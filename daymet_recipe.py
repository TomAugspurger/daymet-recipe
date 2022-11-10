import enum
import datetime
from pangeo_forge_recipes import patterns

from pangeo_forge_recipes.storage import StorageConfig, FSSpecTarget, CacheFSSpecTarget
from pangeo_forge_recipes.recipes import XarrayZarrRecipe

# import fsspec.implementations
import adlfs
import pandas as pd
import numpy as np
import dask.array as da
import urllib.request
import xarray as xr
import tempfile
import logging
import azure.storage.blob
import rich.logging
import distributed
import dask
import dask_gateway
import zarr


logger = logging.getLogger("daymet")
logger.setLevel(logging.INFO)
logger.addHandler(rich.logging.RichHandler())


class Region(str, enum.Enum):
    NA = "na"
    PR = "pr"
    HI = "hi"


class Frequency(str, enum.Enum):
    DAY = "daily"
    MONTH = "mon"
    YEAR = "ann"


AGG_VARIABLES = {"prcp", "swe", "tmax", "tmin", "vp"}
DAILY_VARIABLES = AGG_VARIABLES | {"dayl", "srad"}
DATES = [datetime.datetime(y, 1, 1) for y in range(1980, 2022)]


def make_format_function(region, frequency):
    region = Region(region)
    frequency = Frequency(frequency)

    if frequency in {Frequency.MONTH, Frequency.YEAR}:

        def format_function(variable, time):
            # https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/1855/daymet_v4_prcp_monttl_hi_1980.nc
            assert variable in AGG_VARIABLES

            folder = "2130" if frequency == Frequency.YEAR else "2131"
            if variable == "prcp":
                agg = "ttl"
            else:
                agg = "avg"
            return f"https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/{folder}/daymet_v4_{variable}_{frequency}{agg}_{region}_{time:%Y}.nc"

    else:

        def format_function(variable, time):
            assert variable in DAILY_VARIABLES
            # https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/1840/daymet_v4_daily_hi_dayl_1980.nc
            return f"https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/2129/daymet_v4_{frequency}_{region}_{variable}_{time:%Y}.nc"

    return format_function


class MyFSSpecTarget(FSSpecTarget):
    def __post_init__(self):
        pass


class MyCacheFSSpecTarget(CacheFSSpecTarget):
    def __post_init__(self):
        pass


def make_recipe(region, frequency, credential):
    if frequency == Frequency.DAY:
        variables = list(DAILY_VARIABLES)
        nitems_per_file = 365
        kwargs = dict(subset_inputs={"time": 365})
    else:
        variables = list(AGG_VARIABLES)

        if frequency == Frequency.YEAR:
            nitems_per_file = 1
            kwargs = dict()
        else:
            nitems_per_file = 12
            kwargs = dict(subset_inputs={"time": 12})

    variable_merge_dim = patterns.MergeDim("variable", keys=variables)

    dates = DATES
    concat_dim = patterns.ConcatDim("time", keys=dates, nitems_per_file=nitems_per_file)

    pattern = patterns.FilePattern(
        make_format_function(region, frequency), variable_merge_dim, concat_dim
    )

    fs = adlfs.AzureBlobFileSystem(
        "daymeteuwest",
        credential=credential,
    )
    storage_config = StorageConfig(
        target=MyFSSpecTarget(fs, root_path=f"test-update/{frequency}/{region}.zarr"),
        cache=MyCacheFSSpecTarget(
            fs, root_path=f"test-update/cache/{frequency}/{region}"
        ),
    )

    recipe = XarrayZarrRecipe(
        pattern, storage_config=storage_config, copy_input_to_local_file=True, **kwargs
    )

    return recipe


def make_year(recipe, year, freq):
    datasets = {}
    if freq == Frequency.DAY:
        variables = DAILY_VARIABLES
    else:
        variables = AGG_VARIABLES
    for variable in variables:
        with tempfile.NamedTemporaryFile() as tf:
            url = recipe.file_pattern.format_function(variable, year)
            logger.info("Downloading %s", url)
            filename, _ = urllib.request.urlretrieve(url, filename=tf)
            datasets[variable] = xr.open_dataset(filename)

    ds = xr.merge(list(datasets.values()), join="exact", combine_attrs="no_conflicts")
    return ds


def make_full_template(recipe, tpl, frequency):
    coords = {"lat": tpl.lat, "lon": tpl.lon, "x": tpl.x, "y": tpl.y}
    if frequency == Frequency.YEAR:
        time = xr.DataArray(
            pd.date_range("1980-07-01T12:00:00", periods=len(DATES), freq="AS-JUL"),
            name=tpl.time.name,
            dims=tpl.time.dims,
            attrs=tpl.time.attrs,
        )
    elif frequency == Frequency.MONTH:
        arrays = [tpl.time + pd.Timedelta(days=365 * i) for i in range(len(DATES))]
        time = xr.DataArray(
            np.concatenate(arrays),
            name=tpl.time.name,
            dims=tpl.time.dims,
            attrs=tpl.time.attrs,
        )
    else:
        arrays = [
            pd.date_range(f"{date:%Y}-01-01T12:00:00", freq="D", periods=365)
            for date in DATES
        ]
        time = xr.DataArray(
            np.concatenate(arrays),
            name=tpl.time.name,
            dims=tpl.time.dims,
            attrs=tpl.time.attrs,
        )

    coords["time"] = time
    attrs = dict(tpl.attrs)

    data_vars = {}
    for k, v in tpl.data_vars.items():
        if "time" in v.dims:
            shape = (recipe.nitems_per_input * len(DATES),) + v.shape[1:]
            data = da.zeros(shape, chunks=(1,) + ("auto",) * len(v.shape[1:]))
            data_vars[k] = xr.DataArray(
                data,
                coords={k: coords[k] for k in v.coords},
                dims=v.dims,
                attrs=v.attrs,
                name=v.name,
            )
        else:
            data_vars[k] = v
    full = xr.Dataset(data_vars, coords, attrs=attrs)
    return full


def prepare(recipe, frequency, store):
    tpl = make_year(recipe, DATES[0], frequency)
    full = make_full_template(recipe, tpl, frequency)
    full.to_zarr(store, mode="w", compute=False)
    return full


def write_year(recipe, frequency, date, i, store):
    year = make_year(recipe, date, frequency)
    step = recipe.nitems_per_input
    zarr_region = {"time": slice(i * step, (i + 1) * step)}
    year.drop_vars(["lambert_conformal_conic", "lat", "lon", "x", "y"]).to_zarr(
        store, region=zarr_region
    )


def is_initialized(store, variable):
    try:
        info = zarr.open_consolidated(store)[variable].info
    except (azure.core.exceptions.ResourceNotFoundError, KeyError):
        return False
    return info.obj.nchunks_initialized >= info.obj.nchunks


def get_cluster():
    dask.config.set({"distributed.comm.timeouts.connect": "300s"})

    env = {"DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT": "300s"}

    gateway = dask_gateway.Gateway()
    options = gateway.cluster_options()
    options.environment.update(env)
    cluster = gateway.new_cluster(options, shutdown_on_close=False)
    cluster.scale(8)
    with cluster.get_client() as client:
        client.upload_file("daymet_recipe.py")
    return cluster


def run_one(
    region,
    frequency,
    credential,
    account_url="https://daymeteuwest.blob.core.windows.net",
    container_name="test-update",
    cluster=None,
):
    if cluster is None:
        close_cluster = True
        cluster = get_cluster()
    else:
        close_cluster = False
    client = cluster.get_client()

    store = zarr.storage.ABSStore(
        client=azure.storage.blob.ContainerClient(
            account_url, container_name, credential
        ),
        prefix=f"fix/{frequency}-{region}.zarr",
    )
    recipe = make_recipe(region, frequency, credential)
    variables = DAILY_VARIABLES if frequency == Frequency.DAY else AGG_VARIABLES

    r = client.gather(
        [client.submit(is_initialized, store=store, variable=v) for v in variables]
    )
    if all(r):
        print(f"{region}-{frequency} - already initialized")
        return store

    print(f"{region}-{frequency} - creating template")
    full_ = client.submit(prepare, recipe, frequency, store)
    full_.result()
    # display(full)

    print(f"{region}-{frequency} - writing years")
    futures = [
        client.submit(
            write_year, recipe=recipe, frequency=frequency, date=date, i=i, store=store
        )
        for i, date in enumerate(DATES)
    ]
    distributed.fire_and_forget(futures)
    _ = distributed.wait(futures)
    r = client.gather(
        [client.submit(is_initialized, store=store, variable=v) for v in variables]
    )
    assert all(r)

    if close_cluster:
        client.close()
        cluster.close()
    return store
