PKGS=(
    libgl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols 
    libegl1-mesa-dev libc++-dev libglew-dev libeigen3-dev cmake g++ 
    ninja-build libjpeg-dev libpng-dev libavcodec-dev libavutil-dev 
    libavformat-dev libswscale-dev libavdevice-dev
)

for pkg in "${PKGS[@]}"; do
    installed=$(dpkg-query -W --showformat='${Status}\n' $pkg 2>/dev/null | grep "install ok installed")
    if [ "" != "$installed" ]; then
        echo -e "$pkg\t[\033[32m已安装\033[0m]"
    else
        echo -e "$pkg\t[\033[31m未安装\033[0m]"
    fi
done