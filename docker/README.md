# Docker support
> Virtualization technology helps to make end users happy, while killing those people who take on deployment responsibility at the same time. :smirk:

Tensorflow is mainly targeted at Ubuntu 16.04 with specific Python/Driver/CUDA/CUDNN/C++ versions - lots of mysteries in the early days. So often you can stuck into a situation where non of the official pre-build package fits your computer, especially when Ubuntu 18.04 was just released and you accidentally decided to upgrade too early.

In the end, I am tired of configuration and compilation (sometimes not even working) due to these version conflict issues, and decided to build my own docker image stick to the officially recommended system requirements.

But docker does not support depth camera (USB driver issue), so it cannot be used by the tracking part. Check my other repo '[docker-hub](https://github.com/xkunwu/docker-hub)' for a much better maintained general purpose docker.
