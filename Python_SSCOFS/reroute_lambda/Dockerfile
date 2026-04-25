# AWS Lambda container for the on-demand race reroute.
#
# Build context: REPO ROOT (so the polar CSV at the repo root is reachable).
#
#   docker build \
#     -f OceanCurrents/Python_SSCOFS/reroute_lambda/Dockerfile \
#     -t race-reroute:latest .
#
# Why a container, not zip: the sail_routing stack pulls in numpy / scipy /
# xarray / pyproj / numba / matplotlib + their native deps. Easily blows
# past Lambda's 250 MB unzipped layer cap. A container image (10 GB cap)
# fits comfortably (~600-800 MB).
FROM public.ecr.aws/lambda/python:3.11

# --- Build toolchain ---
# Several transitive deps (contourpy, h5py, pyproj on bleeding-edge versions)
# fall back to source builds when pip can't find a wheel. Without a C/C++
# compiler in the image, those source builds fail with "Unknown compiler"
# errors. Amazon Linux 2023 ships dnf; gcc-c++ + python3-devel cover all
# observed cases. Adds ~150 MB to the image; acceptable for a 600-800 MB
# baseline. Removed from the runtime image via two-stage build later if
# we ever need to trim.
RUN yum install -y gcc gcc-c++ python3-devel && yum clean all

# --- Python dependencies ---
# We deliberately don't reuse OceanCurrents/Python_SSCOFS/requirements.txt:
# the GHA workflow pins a slightly different working set that the Lambda
# doesn't need at runtime. Keep this list minimal — bump alongside
# requirements.txt when routing-relevant deps move.
# Pin to versions with prebuilt manylinux py3.11 amd64 wheels. numpy 2.4
# (Oct 2026) is source-only on py3.11, so unconstrained installs land us
# in meson + gcc territory inside the image. The pinned set below mirrors
# what the GHA precompute workflow ends up resolving on ubuntu-latest.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "numpy>=1.26,<2.3" \
      "scipy>=1.11,<1.14" \
      "numba>=0.59,<0.62" \
      "h5py>=3.10,<3.13" \
      "pandas<3" \
      "xarray<2025" \
      "h5netcdf>=1.3,<1.7" \
      s3fs \
      boto3 \
      requests \
      PyYAML \
      "pyproj>=3.6,<3.7"
# matplotlib intentionally NOT installed: only the plot-only paths in
# run_route.py / plot_local_currents.py touch it, and Lambda never runs
# those (race-mode forces --no-plots). Saves ~150 MB and avoids the
# /home/sbx_user1051/.config/matplotlib mkdir warnings.

WORKDIR ${LAMBDA_TASK_ROOT}

# Router core + helpers (paths are repo-root-relative; see build invocation).
COPY OceanCurrents/Python_SSCOFS/sail_routing.py        ${LAMBDA_TASK_ROOT}/
COPY OceanCurrents/Python_SSCOFS/run_route.py           ${LAMBDA_TASK_ROOT}/
COPY OceanCurrents/Python_SSCOFS/race_publish.py        ${LAMBDA_TASK_ROOT}/
COPY OceanCurrents/Python_SSCOFS/ecmwf_wind.py          ${LAMBDA_TASK_ROOT}/
COPY OceanCurrents/Python_SSCOFS/sscofs_cache.py        ${LAMBDA_TASK_ROOT}/
COPY OceanCurrents/Python_SSCOFS/plot_local_currents.py ${LAMBDA_TASK_ROOT}/
COPY OceanCurrents/Python_SSCOFS/water_boundary.py      ${LAMBDA_TASK_ROOT}/
COPY OceanCurrents/Python_SSCOFS/shoreline_utils.py     ${LAMBDA_TASK_ROOT}/
COPY OceanCurrents/Python_SSCOFS/build_nav_mask.py      ${LAMBDA_TASK_ROOT}/
COPY OceanCurrents/Python_SSCOFS/latest_cycle.py        ${LAMBDA_TASK_ROOT}/
COPY OceanCurrents/Python_SSCOFS/fetch_sscofs.py        ${LAMBDA_TASK_ROOT}/

# Polar table: the YAML's relative path (`../../j105_new_polars.csv`)
# resolves outside the Lambda root, so handler.py rewrites the YAML to
# this absolute path at request time.
COPY j105_new_polars.csv                                ${LAMBDA_TASK_ROOT}/j105_new_polars.csv

# Race YAMLs the Lambda is allowed to serve. Add new races here.
COPY OceanCurrents/Python_SSCOFS/routes/races           ${LAMBDA_TASK_ROOT}/routes/races

# Static data (shoreline polygons, nav mask, etc.) the router reads at solve
# time. The data dir is a few MB; bake it in.
COPY OceanCurrents/Python_SSCOFS/data                   ${LAMBDA_TASK_ROOT}/data

# Lambda entrypoint
COPY OceanCurrents/Python_SSCOFS/reroute_lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py

CMD ["handler.handler"]
