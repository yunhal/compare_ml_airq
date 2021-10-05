find . -name '*.pyc' -delete
find . -name '*.pyo' -delete
find . -name '.DS_Store' -delete

# update info

git config --global credential.helper store

git config --global user.email "yunha.lee.00@gmail.com"
git config --global user.name "yunhal"

git add --all
git commit -m "edit"

git pull
git push
