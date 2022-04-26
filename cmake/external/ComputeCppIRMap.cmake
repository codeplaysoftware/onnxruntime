#/***************************************************************************
# *
# *  @license
# *  Copyright (C) Codeplay Software Limited
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  For your convenience, a copy of the License has been included in this
# *  repository.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  @filename ComputeCppIRMap.cmake
# *
# **************************************************************************/

cmake_minimum_required(VERSION 3.4.3)

# These should match the types of IR output by compute++
set(IR_MAP_spir bc)
set(IR_MAP_spir64 bc)
set(IR_MAP_spir32 bc)
set(IR_MAP_spirv spv)
set(IR_MAP_spirv64 spv)
set(IR_MAP_spirv32 spv)
set(IR_MAP_aorta-x86_64 o)
set(IR_MAP_aorta-aarch64 o)
set(IR_MAP_aorta-rcar-cve o)
set(IR_MAP_custom-spir64 bc)
set(IR_MAP_custom-spir32 bc)
set(IR_MAP_custom-spirv64 spv)
set(IR_MAP_custom-spirv32 spv)
set(IR_MAP_ptx64 s)
set(IR_MAP_amdgcn s)
