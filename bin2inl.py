#
# Copyright(c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http: // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright(c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import sys


def main():
    with open(sys.argv[1], "rb") as input, open(sys.argv[2], "w") as output:
        while (b := input.read(4)):
            output.write("0x{:x}, ".format(
                (b[3] << 24) | (b[2] << 16) | (b[1] << 8) | b[0]))


if __name__ == "__main__":
    main()
