# -*- coding: utf-8 -*-
"""
テストデータの空中写真の色合いを、教師データの色合いに調整するスクリプト（GUI入力対応）

Author: 荒木光一
Date: 2025-09-10
Version: 1.0.0
License: MIT
"""
import os, cv2, glob
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

def ReadImg(Path):
    """
    画像ファイルを読み込み、OpenCV形式の画像として返す。

    Args:
        path (str): 画像ファイルのパス。

    Returns:
        numpy.ndarray: 読み込んだ画像データ（グレースケール）。
    """
    with open(Path, 'rb') as f:
        Data = np.frombuffer(f.read(), np.uint8)

    return cv2.imdecode(Data, cv2.IMREAD_GRAYSCALE)

def WriteImg(Path, ImgData):
    """
    画像データを指定パスに保存する。

    Args:
        path (str): 保存先のファイルパス。
        img (numpy.ndarray): 保存する画像データ。

    Returns:
        bool: 保存に成功した場合はTrue、失敗した場合はFalse。
    """
    Ext = os.path.splitext(Path)[1]
    _, encoded_img = cv2.imencode(Ext, ImgData)

    with open(Path, mode='wb') as f:
        encoded_img.tofile(f)

def CalcMeanStd(ImgPath):
    """
    指定した画像ファイルの平均値と標準偏差、および画像データを返す。

    Args:
        ImgPath (str): 画像ファイルのパス。

    Returns:
        tuple:
            float: 画像の平均値。
            float: 画像の標準偏差。
            numpy.ndarray: 画像データ（グレースケール）。
    """
    ImgData = ReadImg(ImgPath)
    Mean, Std = cv2.meanStdDev(ImgData)

    return float(Mean), float(Std), ImgData

def GetImgMeanStd(ImgDir, ImageExtension):
    """
    指定フォルダ内の全画像の平均値と標準偏差の平均を計算する。

    Args:
        ImgDir (str): 画像フォルダのパス。
        ImageExtension (str): 画像拡張子（例: .png）。

    Returns:
        tuple:
            float: フォルダ内画像の平均値の平均。
            float: フォルダ内画像の標準偏差の平均。
    """
    MeanList, StdList = [], []
    for ImgPath in glob.glob(os.path.join(ImgDir, '*' + ImageExtension)):

        Mean, Std, _ = CalcMeanStd(ImgPath)
        MeanList.append(Mean)
        StdList.append(Std)

    return np.mean(MeanList), np.mean(StdList)

def AdjustImage(TestImgData, TestMean, TestStd, TrainMean, TrainStd):
    """
    テスト画像の明るさ・コントラストを教師画像の統計値に合わせて調整する。

    Args:
        TestImgData (numpy.ndarray): 調整対象の画像データ。
        TestMean (float): テスト画像の平均値。
        TestStd (float): テスト画像の標準偏差。
        TrainMean (float): 教師画像群の平均値。
        TrainStd (float): 教師画像群の標準偏差。

    Returns:
        numpy.ndarray: 調整後の画像データ（uint8型）。
    """
    TestImgData = TestImgData.astype(np.float32)
    TestImgData = (TestImgData - TestMean) / (TestStd + 1e-8) * TrainStd + TrainMean
    TestImgData = np.clip(TestImgData, 0, 255)

    return TestImgData.astype(np.uint8)

def DrawBlackBasedThreshold(ImgData, Threshold):
    """
    指定した閾値未満の画素値を0（黒）にする。

    Args:
        ImgData (numpy.ndarray): 入力画像データ。
        Threshold (int or float): 閾値。

    Returns:
        numpy.ndarray: 閾値処理後の画像データ。
    """
    Result = ImgData.copy()
    Result[Result < Threshold] = 0

    return Result

def Adjust(TrainImgDir, TestImgDir, ImageExtension, OutputDir, Threshold):
    """
    テスト画像の明るさ・コントラストを教師画像群の統計値に合わせて調整し、閾値処理後に出力フォルダへ保存する。

    Args:
        TrainImgDir (str): 教師画像フォルダのパス。
        TestImgDir (str): テスト画像フォルダのパス。
        ImageExtension (str): 画像拡張子（例: .png）。
        OutputDir (str): 出力フォルダのパス。
        Threshold (int): 閾値（この値未満の画素は0にする）。
    """
    TrainMean, TrainStd = GetImgMeanStd(TrainImgDir, ImageExtension)
    print('Train Dir: Mean\t', TrainMean, '', TrainStd)

    TestImgPathList = glob.glob(os.path.join(TestImgDir, '*' + ImageExtension))
    for TestImgPath in TestImgPathList:

        TestMean, TestStd, TestImgData = CalcMeanStd(TestImgPath)
        AdjImg = AdjustImage(TestImgData, TestMean, TestStd, TrainMean, TrainStd)
        AdjImg = DrawBlackBasedThreshold(AdjImg, Threshold)

        BaseFilename = os.path.splitext(os.path.basename(TestImgPath))[0]
        OutputPath = os.path.join(OutputDir, BaseFilename + '.png')
        WriteImg(OutputPath, AdjImg)

def SelectDir(Entry):
    """
    ディレクトリ選択ダイアログを開き、選択結果をエントリに反映する。

    Args:
        entry (tk.Entry): パスを入力するエントリウィジェット。
    """
    Path = filedialog.askdirectory()

    if Path:
        Entry.delete(0, tk.END)
        Entry.insert(0, Path)

def OnRun():
    """
    GUIの「実行」ボタン押下時の処理。入力値を取得し推論を実行する。
    """
    try:
        TrainImgDir = EntryTrainImgDir.get()
        TestImgDir = EntryTestImgDir.get()
        ImageExtension = EntryExt.get()
        ThresholdStr = EntryThreshold.get()
        OutputDir = EntryOutputDir.get()

        if not (TrainImgDir and TestImgDir and ImageExtension and ThresholdStr and OutputDir):
            messagebox.showerror('エラー', 'すべての項目を入力してください')
            return

        Adjust(TrainImgDir, TestImgDir, ImageExtension, OutputDir, int(ThresholdStr))
        messagebox.showinfo('完了', '処理が完了しました')

    except Exception as e:
        messagebox.showerror('エラー', str(e))

Root = tk.Tk()
Root.title('テストデータの色調整（Ver. 1.0.0）')

tk.Label(Root, text='教師データの画像フォルダ').grid(row=0, column=0, sticky='e')
EntryTrainImgDir = tk.Entry(Root, width=40)
EntryTrainImgDir.grid(row=0, column=1)
tk.Button(Root, text='参照', command=lambda: SelectDir(EntryTrainImgDir)).grid(row=0, column=2)

tk.Label(Root, text='テストデータの画像フォルダ').grid(row=1, column=0, sticky='e')
EntryTestImgDir = tk.Entry(Root, width=40)
EntryTestImgDir.grid(row=1, column=1)
tk.Button(Root, text='参照', command=lambda: SelectDir(EntryTestImgDir)).grid(row=1, column=2)

tk.Label(Root, text='画像の拡張子（例:png）').grid(row=2, column=0, sticky='e')
EntryExt = tk.Entry(Root, width=20)
EntryExt.grid(row=2, column=1, sticky='w')

tk.Label(Root, text='閾値（整数）').grid(row=3, column=0, sticky='e')
EntryThreshold = tk.Entry(Root, width=10)
EntryThreshold.insert(0, '128')
EntryThreshold.grid(row=3, column=1, sticky='w')

tk.Label(Root, text='出力フォルダ').grid(row=4, column=0, sticky='e')
EntryOutputDir = tk.Entry(Root, width=40)
EntryOutputDir.grid(row=4, column=1)
tk.Button(Root, text='参照', command=lambda: SelectDir(EntryOutputDir)).grid(row=4, column=2)

tk.Button(Root, text='実行', command=OnRun, width=20).grid(row=5, column=0, columnspan=3, pady=10)

Root.mainloop()
