#!/bin/bash
set -ex

BUILD_DIR=$PWD/cmake-build-debug
SO_DEST=$BUILD_DIR/libtipscore.so

[ ! -f ${SO_DEST} ] && ln -s $BUILD_DIR/libtipscore.so tips/tensorflow/

cd $BUILD_DIR
export PYTHONPATH="${PYTHONPATH}:$PWD"

ctest -V
