# 4. ModelScope 数据集上传与下载

数据集仓库为 [`MaZp001/Bolt`](https://modelscope.cn/datasets/MaZp001/Bolt)。

## 4.1 安装并登录

```bash
python3 -m pip install --upgrade modelscope
modelscope login --token ms-********************************
```

执行 `modelscope login` 后，根据提示输入 ModelScope Access Token。请勿将 Token 直接写入 README 或提交到 Git 仓库。

## 4.2 上传
```bash
modelscope upload 仓库名 本地文件 仓库中的路径
```
以下命令会递归上传当前路径下下的全部文件和子目录，并保存到数据集仓库根目录：

当前目录:
```bash
modelscope upload MaZp001/Bolt . --repo-type model
```
制定目录:
```bash
modelscope upload MaZp001/Bolt ./logs --repo-type model
```
指定文件:
```bash
modelscope upload MaZp001/Bolt \
    ~/Bolt/logs/README.md \
    README.md \
    --repo-type model
```

再次执行该命令可以上传新增或修改后的文件。

## 4.3 下载

以下命令默认下载到当前目录：

```bash
modelscope download MaZp001/Bolt --repo-type model --local-dir .
```

如果需要直接同步到当前 `data` 目录，可以将 `--local-dir` 后的路径改为 ``。执行前请确认本地同名文件可以被更新。