#!/bin/bash

# stop if error
set -e

read -p 'Release version: ' version

GITHUB_PAT="$( awk -F'=' '/^GITHUB_PAT/ { print $2}' .env)"
pipyUser="$( awk -F'=' '/^pipyUser/ { print $2}' .env)"
pipyPassword="$( awk -F'=' '/^pipyPassword/ { print $2}' .env)"

if [ -z "${GITHUB_PAT}" ]; then
  read -p 'You personal access token for Github: ' GITHUB_PAT
fi

if [ -z "${pipyUser}" ]; then
  read -p 'Your username for pipy: ' pipyUser
fi

if [ -z "${pipyPassword}" ]; then
  read -p 'Your password for pipy: ' pipyPassword
fi


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
git checkout release/${version}

# Tagging proper version
echo "Tagging proper version"
git tag v${version}

# Build release
echo "Building latest build"
make install stop
docker-compose run --rm -T john_dev poetry build
#docker run --rm -v ${PWD}:/work -w /work john-toolbox:latest poetry build

echo "Merging into develop and main"
git checkout main
git merge --no-ff origin/release/${version} -m "chore: release v${version} (merge)"
git checkout develop
git merge --no-ff origin/release/${version} -m "chore: release v${version} (merge)"

echo "Pushing branch"
git push origin develop
git push origin main
echo "Pushing tag"
git push origin --tags

echo "Making github release"
docker run -v ${PWD}:/work -w /work --entrypoint "" release-changelog:latest conventional-github-releaser -p angular --token ${GITHUB_PAT}

# Build release
echo "Building latest build"
docker-compose run --rm -T john_dev poetry build
#docker run --rm -v ${PWD}:/work -w /work john-toolbox:latest poetry build
# Build release
echo "Publishing latest build"
docker-compose run --rm -T john_dev poetry publish -u ${pipyUser} -p ${pipyPassword}
#docker run --rm -v ${PWD}:/work -w /work john-toolbox:latest poetry publish -u ${pipyUser} -p ${pipyPassword}

echo "Deleting release branch"
git checkout develop
git push origin :release/${version}

cd ${localDir}
sudo chmod -R 777 ${releaseDir}
rm -rf ${releaseDir}