# 数字图像处理课程作业

**姓名：** 席越  
**学号：** BZ25001010

本仓库作为数字图像处理课程作业的统一提交仓库。各次作业按照 `Assignment_xx` 的形式建立独立子文件夹，便于查看代码、运行说明、实验结果和报告内容。

## 仓库目录结构

```text
Digital_image_Homework/
├── README.md
├── Assignment_01/
│   ├── README.md
│   ├── requirements.txt
│   └── code/
├── Assignment_02/
│   ├── README.md
│   ├── requirements.txt
│   └── hw_2/
├── Assignment_03/
│   ├── README.md
│   ├── requirements.txt
│   ├── code/
│   ├── data/
│   └── outputs/
└── Assignment_04/
    ├── README.md
    ├── requirements.txt
    ├── code/
    ├── data/chair/images/
    ├── pics/
    └── outputs/
```

## 作业列表

| 作业 | 主题 | 代码位置 | 说明文档 |
|---|---|---|---|
| Assignment 01 | 图像几何变换与图像形态学 | `Assignment_01/code/` | `Assignment_01/README.md` |
| Assignment 02 | Poisson 图像编辑与 Pix2Pix 图像翻译 | `Assignment_02/hw_2/` | `Assignment_02/README.md` |
| Assignment 03 | Bundle Adjustment 与 COLMAP 3D Reconstruction | `Assignment_03/code/` | `Assignment_03/README.md` |
| Assignment 04 | Simplified 3D Gaussian Splatting | `Assignment_04/code/` | `Assignment_04/README.md` |

## 通用运行方式

每次作业均在对应文件夹下提供依赖文件和运行说明。例如：

```bash
cd Assignment_04
pip install -r requirements.txt
```

具体运行命令、输入输出说明、实验结果和分析见每个作业文件夹中的 `README.md`。

## 提交说明

每次作业单独放在一个子文件夹中，报告内容采用类似论文项目说明的形式，通常包含：

1. 作业目标与方法简介；
2. 环境依赖与安装方式；
3. 运行脚本；
4. 输入和输出说明；
5. 测试结果与分析；
6. 小结；
7. 参考资料。
