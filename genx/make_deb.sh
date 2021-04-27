#!/bin/sh

# exit on error
set -e

python setup.py bdist_rpm

# These commands requires dpkg-dev alien
echo "Moving distribution files..."

cd dist

NAME=$(basename *.tar.gz .tar.gz)

echo "Creating debian folder..."
fakeroot alien -k -g "${NAME}-1.noarch.rpm"

# creating menu entries
mkdir -p "${NAME}/usr/share/applications/"
mkdir -p "${NAME}.orig/usr/share/applications/"
cp ../debian_build/*.desktop "${NAME}/usr/share/applications/"
cp ../debian_build/*.desktop "${NAME}.orig/usr/share/applications/"


# Icons
#   menu
mkdir -p "${NAME}/usr/share/pixmaps/"
mkdir -p "${NAME}.orig/usr/share/pixmaps/"
cp ../debian_build/*.xpm "${NAME}/usr/share/pixmaps/"
cp ../debian_build/*.xpm "${NAME}.orig/usr/share/pixmaps/"

#   mime
mkdir -p "${NAME}/tmp/genx_icons"
mkdir -p "${NAME}.orig/tmp/genx_icons"
cp ../debian_build/genx_*.png "${NAME}/tmp/genx_icons"
cp ../debian_build/genx_*.png "${NAME}.orig/tmp/genx_icons"


# creating mime types
mkdir -p "${NAME}/usr/share/mime/packages/"
mkdir -p "${NAME}.orig/usr/share/mime/packages/"
cp ../debian_build/*.xml "${NAME}/usr/share/mime/packages/"
cp ../debian_build/*.xml "${NAME}.orig/usr/share/mime/packages/"


cd ${NAME}
cp ../../debian_build/control debian/
cp ../../debian_build/postinst debian/
cp ../../debian_build/postrm debian/

# create .deb package

PY_VERSION=$(python -c "import sys; print(sys.version_info.major + sys.version_info.minor)")

dpkg-buildpackage -i.* -I -rfakeroot -us -uc > ../last_package.log
cd ..

#os.rename((__name__+'_'+__version__).lower()+'-1_all.deb', __name__+'-'+__version__+'_py%i%i.deb'%(
#                    py_version.major,py_version.minor))

echo "Removing debian folder"

rm -r ${NAME}
rm *.orig.tar.gz
rm *.rpm


echo "Removing build folder..."
cd ..
rm -r build
