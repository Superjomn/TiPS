#pragma once

// This file contains some operations those will expose to python.

namespace tips {

extern "C" {
//! Initialize TiPS service.
void tips_init();

//! Shutdown the whole TiPS service.
void tips_shutdown();

bool tips_is_initialize();

//! Get the number of nodes.
int tips_size();

//! Get the rank of the node.
int tips_rank();
}

}  // namespace tips
