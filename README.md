<h1 align="center">

Welcome to john_toolbox 👋

</h1>


![Version](https://img.shields.io/badge/version-0.5.1-blue.svg?cacheSeconds=2592000)
[![Documentation](https://img.shields.io/badge/documentation-yes-brightgreen.svg)](https://nguyenanht.github.io/john-toolbox/)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)
![Licence](https://img.shields.io/badge/License-MIT-FFB600.svg)
[![Downloads](https://static.pepy.tech/personalized-badge/john-toolbox?period=month&units=international_system&left_color=grey&right_color=red&left_text=Downloads/Month)](https://pepy.tech/project/john-toolbox)
[![Downloads](https://static.pepy.tech/personalized-badge/john-toolbox?period=total&units=international_system&left_color=grey&right_color=red&left_text=Downloads/Total)](https://pepy.tech/project/john-toolbox)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://github.com/scikit-learn/scikit-learn)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://github.com/pandas-dev/pandas)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://github.com/numpy/numpy)
[![Poetry](https://img.shields.io/badge/poetry-%233B82F6.svg?style=for-the-badge&logo=poetry&logoColor=white)](https://github.com/python-poetry/poetry)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://github.com/pytorch/pytorch)

> This is my own toolbox to handle preprocessing production ready based on scikit-learn Pipeline but with more flexibility.
### 🏠 [Homepage](https://github.com/nguyenanht/john-toolbox)

# 💿 Installation with pip
```sh
pip install john-toolbox
```

# 💡 How to use the package ?

If you want examples, please refer to [notebooks directory](https://github.com/nguyenanht/john-toolbox/tree/develop/notebooks). It contains tutorials on how to use the package and other useful tutorials to handle end to end machine learning project.

# 🚧 Local development
## 💣 Installation guide
❗ by default, we install docker container with cpu only, if you want to install gpu mode :
```sh
make env
```
Comment in `.env` file at the root, the following lines :
```dotenv
DEVICE=cpu
DOCKER_RUNTIME=runc
```
Uncomment in `.env` file at the root, the following lines :
```dotenv
# DEVICE=gpu
# DOCKER_RUNTIME=nvidia
```
then :
```sh
make install
```
if you want to use with local domain name, you need to generate certificate ssl in local development instead of url with port like `http://localhost:8885` :
```sh
make stop ssl
```
## ✨ Usage
### Start project :
```sh
make start
```
### Stop project : 
```sh
make stop
```
### Display logs of specific service
```sh
make logs svc="your_service_name_declared_in_docker_compose"
```

### Go inside a docker container
```sh
./cli your_service_name_declared_in_docker_compose
```

### 🦸 Need help
```sh
make help
```
### Url
- Develop with Jupyter Notebook : https://nb.johntoolbox.localhost
- Treafik (reverse proxy) : https://proxy.johntoolbox.localhost

## Author
👤 **Johnathan Nguyen**
* GitHub: [@nguyenanht](https://github.com/{github_username})


# Show your support
Give a ⭐️ if this project helped you!

# 🤝 Contributing
Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/nguyenanht/john-toolbox/issues).
## How to contribute
### Semantic Commit Messages

Every programmer in this project must respect a convention for every commit.
The CI will not let you merge your branch into Develop.

See how a minor change to your commit message style can make you a better programmer.

Format: `<type>(<scope>): <subject>`

`<scope>` is optional

#### Example

```
feat: add hat wobble
^--^  ^------------^
|     |
|     +-> Summary in present tense.
|
+-------> Type: chore, docs, feat, fix, refactor, style, or test.
```

More Examples:

- `feat`: (new feature for the user, not a new feature for build script)
- `fix`: (bug fix for the user, not a fix to a build script)
- `docs`: (changes to the documentation)
- `style`: (formatting, missing semicolons, etc.; no production code change)
- `refactor`: (refactoring production code, e.g. renaming a variable)
- `test`: (adding missing tests, refactoring tests; no production code change)
- `chore`: (updating grunt tasks etc.; no production code change)

References:

- https://www.conventionalcommits.org/
- https://seesparkbox.com/foundry/semantic_commit_messages
- http://karma-runner.github.io/1.0/dev/git-commit-msg.html

# Useful link
- how to publish new version in pypi with poetry ? : https://johnfraney.ca/posts/2019/05/28/create-publish-python-package-poetry/
- how to create a new release ? : https://www.atlassian.com/fr/git/tutorials/comparing-workflows/gitflow-workflow
- how to generate docs : https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion
- how to deploy with github actions : https://blog.flozz.fr/2020/09/21/deployer-automatiquement-sur-github-pages-avec-github-actions/

---
_This README was created with the [markdown-readme-generator](https://github.com/pedroermarinho/markdown-readme-generator)_