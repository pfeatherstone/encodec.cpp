![Ubuntu](https://github.com/pfeatherstone/encodec.cpp/actions/workflows/ubuntu.yml/badge.svg)
<!-- ![MacOS](https://github.com/pfeatherstone/encodec.cpp/actions/workflows/macos.yml/badge.svg)
![Windows](https://github.com/pfeatherstone/encodec.cpp/actions/workflows/windows.yml/badge.svg) -->

# encodec.cpp

A C++ implementation of Meta's [Encodec](https://audiocraft.metademolab.com/encodec.html) codec using [Eigen](https://gitlab.com/libeigen/eigen).
The weights are stored inside the library.

## API

```cpp
encodec::encoder enc;
encodec::decoder dec;

float audio[24000];
size_t bps = 24000; // 12000, 6000 or 3000
std::span<const uint8_t> packet = enc.encode(audio, bps);
std::span<const float>   audio2 = dec.decode(packet, bps);
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

Pretrained weights downloaded by helper scripts are subject to their own licenses. See THIRD_PARTY_NOTICES.md for details.