#!/bin/bash

# stop if error
set -e

read -p 'Release version: ' version
echo ${version} | grep v && echo "Version should be x.y.z (for example, 1.1.1, 2.0.0, ...)" && exit -1

localDir=`readlink -f .`
releaseDir="${localDir}/release-${version}"
rm -rf ${releaseDir}
mkdir ${releaseDir}
cd $releaseDir

echo "Cloning repo into john-toolbox"
git clone -q git@github.com:nguyenanht/john-toolbox.git john-toolbox

cd john-toolbox
# Create release branch and push it
git checkout -b release/${version}
# Change version of package
docker run --rm -v ${PWD}:/work -w /work john-toolbox:latest poetry version ${version}
# Generate docs
make install docs
# Change version of project in readme
sed -e "/version-/{ N; s/version-.*-blue/version-${version}-blue/ }" README.md  > tmp.md
cat tmp.md > README.md
rm tmp.md
# change version of __init__.py
old_version="`grep __version__ john_toolbox/__init__.py | cut -d\   -f3`"
old_version=`echo "$old_version" | cut -d'"' -f 2`
new_version=${version}
sed -e "s/${old_version}/${new_version}/g" john_toolbox/__init__.py > john_toolbox/__init__tmp.py
cat john_toolbox/__init__tmp.py > john_toolbox/__init__.py
rm john_toolbox/__init__tmp.py

# Add modified files
git add pyproject.toml README.md john_toolbox/__init__.py docs/
# Commit release
git commit -m "chore: release v${version}"
# Create tag for changelog generation
git tag v${version}
docker run -v ${PWD}:/work -w /work --entrypoint "" release-changelog:latest conventional-changelog -p angular -i CHANGELOG.md -s -r 0
docker run -v ${PWD}:/work -w /work --entrypoint "" release-changelog:latest chmod 777 CHANGELOG.md
# Removing 4 first line of the file
echo "$(tail -n +4 CHANGELOG.md)" > CHANGELOG.md
# Deleting tag 
git tag -d v${version}
# Adding CHANGELOG to commit
git add CHANGELOG.md
git commit --amend --no-edit
# Push release branch
git push origin release/${version}

cd ${localDir}
sudo rm -rf ${releaseDir}