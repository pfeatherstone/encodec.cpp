![Ubuntu](https://github.com/pfeatherstone/encodec.cpp/actions/workflows/ubuntu.yml/badge.svg)
![MacOS](https://github.com/pfeatherstone/encodec.cpp/actions/workflows/macos.yml/badge.svg)
![Windows](https://github.com/pfeatherstone/encodec.cpp/actions/workflows/windows.yml/badge.svg)

# encodec.cpp

A C++ implementation of Meta's [Encodec](https://audiocraft.metademolab.com/encodec.html) using [Eigen](https://gitlab.com/libeigen/eigen).

## Install

Requirements:
* CMake 3.23 or newer
* A C++20 compiler
* [Git Large File Storage (LFS) ](https://git-lfs.com/)

Using cmake FetchContent:

```cmake
FetchContent_Declare(
  encodec
  GIT_REPOSITORY https://github.com/pfeatherstone/encodec.cpp.git
  GIT_TAG        <tag-or-commit>
  GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(encodec)

target_link_libraries(my_encoder_only_app PRIVATE 
  encodec::encodec 
  encodec::encoder24
  encodec::rvq24)

target_link_libraries(my_decoder_only_app PRIVATE 
  encodec::encodec 
  encodec::decoder24
  encodec::rvq24)
```

Using CPM:

```cmake
CPMAddPackage("gh:pfeatherstone/encodec.cpp#<tag>")

target_link_libraries(my_app PRIVATE 
  encodec::encodec 
  encodec::encoder24
  encodec::decoder24
  encodec::rvq24)
```

The `encodec::encoder24`, `encodec::decoder24` and `encodec::rvq24` targets are compiled weight targets.

## API

```cpp
#include <encodec.h>
#include <encodec_encoder24.h>
#include <encodec_decoder24.h>
#include <encodec_rvq24.h>

encodec::encoder enc(encodec::get_encoder24_weights(), encodec::get_rvq24_weights());
encodec::decoder dec(encodec::get_decoder24_weights(), encodec::get_rvq24_weights());

float audio[24000];
size_t bps = 24000; // 12000, 6000 or 3000
std::span<const uint8_t> packet = enc.encode(audio,  encodec::get_encoded_nquantizers(bps));
std::span<const float>   audio2 = dec.decode(packet, encodec::get_encoded_nquantizers(bps));
```

## Notes

* Model weights are compiled into separate libraries, allowing applications to link only the encoder, decoder, and RVQ weights they require.

* You must manually implement streaming for now. Partition your audio into 1s chunks with 10ms overlap. For decoding, use a linear weighting in the overlap regions.

## Features

- [x] Block based API
- [ ] Streaming API

## License

This project is licensed under the MIT License. See the LICENSE file for details.

Pretrained weights downloaded by helper scripts are subject to their own licenses. See THIRD_PARTY_NOTICES.md for details.