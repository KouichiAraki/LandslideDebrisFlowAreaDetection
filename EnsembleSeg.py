# -*- coding: utf-8 -*-
"""
U-Netで空中写真から崩壊地の裸地をセグメンテーションする推論スクリプト（GUI入力対応）

Author: 荒木光一
Date: 2025-09-10
Version: 1.0.0
License: MIT
"""
import glob, os, cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

def ReadImg(Path):
    """
    画像ファイルを読み込み、OpenCV形式の画像として返す。

    Args:
        path (str): 画像ファイルのパス。

    Returns:
        numpy.ndarray: 読み込んだ画像データ。
    """
    with open(Path, 'rb') as f:
        Data = np.frombuffer(f.read(), np.uint8)

    return cv2.imdecode(Data, cv2.IMREAD_COLOR)

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

def Evaluation(ModelPath, TestImgPathList, ImgSize):
    """
    テスト画像の推論を行う。

    Args:
        ModelPath (str): 学習済みモデルのパス。
        TestImgPathList (list of str): テスト画像のパスリスト。
        ImgSize (tuple): 画像サイズ (高さ, 幅, チャンネル)。

    Returns:
        numpy.ndarray: 各画像の推論結果（one-hot配列）。
    """
    K.clear_session()
    Model = load_model(ModelPath, compile=False)

    LikelihoodList = []
    for TestXPath in TestImgPathList:

        print(TestXPath)
        ImgData = ReadImg(TestXPath)
        ImgData = cv2.resize(ImgData, (ImgSize[1], ImgSize[0]), interpolation=cv2.INTER_NEAREST)
        ImgData = ImgData.astype('float32') / 255.
        PredOneHot = Model.predict(np.array([ImgData]))

        LikelihoodList.append(PredOneHot[0])

    return np.array(LikelihoodList)

def ConvertYtoSegImg(PredOneHot, ImgSize):
    """
    モデル出力(one-hot)をセグメンテーション画像に変換する。

    Args:
        PredOneHot (numpy.ndarray): モデルの出力(one-hot形式)。
        ImgSize (tuple): 画像サイズ (高さ, 幅, チャンネル)。

    Returns:
        numpy.ndarray: セグメンテーション画像。
    """
    PredOneHot = tf.argmax(PredOneHot, axis=-1).numpy() if hasattr(tf.argmax(PredOneHot, axis=-1), 'numpy') else tf.argmax(PredOneHot, axis=-1)

    # 裸地(1)は黒、それ以外は白
    mask = (PredOneHot == 1)
    PredData = np.ones((ImgSize[0], ImgSize[1], ImgSize[2]), dtype=np.uint8) * 255
    PredData[mask] = [0, 0, 0]

    return PredData

def RunInf(TestImgDir, ImageExtension, ModelPath, ImgSize, OutputDir):
    """
    指定されたパラメータで推論を実行し、結果画像を出力する。

    Args:
        TestImgDir (str): テスト画像フォルダのパス。
        ImageExtension (str): 画像拡張子（例: .png）。
        ModelPath (str): 学習済みモデルのパス。
        ImgSize (list of int): 画像サイズ [高さ, 幅, チャンネル]。
        OutputDir (str): 出力フォルダのパス。
    """
    TestImgPathList = glob.glob(os.path.join(TestImgDir, '*' + ImageExtension))
    TestImgPathList.sort()

    AllLikelihoodList = Evaluation(ModelPath, TestImgPathList, tuple(ImgSize))

    for TestXPath, Likelihood in zip(TestImgPathList, AllLikelihoodList):

        ImgData = ReadImg(TestXPath)
        PredImg = ConvertYtoSegImg(Likelihood, tuple(ImgSize))
        PredImg = cv2.resize(PredImg, (ImgData.shape[1], ImgData.shape[0]), interpolation=cv2.INTER_NEAREST)

        BaseFilename = os.path.splitext(os.path.basename(TestXPath))[0]
        OutputPath = os.path.join(OutputDir, BaseFilename + '_Seg.jpg')
        WriteImg(OutputPath, PredImg)

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

def SelectFile(Entry):
    """
    ファイル選択ダイアログを開き、選択結果をエントリに反映する。

    Args:
        entry (tk.Entry): パスを入力するエントリウィジェット。
    """
    Path = filedialog.askopenfilename(filetypes=[('Model files', '*.h5;*.hdf5;*.keras;*.*')])

    if Path:
        Entry.delete(0, tk.END)
        Entry.insert(0, Path)

def OnRun():
    """
    GUIの「実行」ボタン押下時の処理。入力値を取得し推論を実行する。
    """
    try:
        TestImgDir = EntryImgDir.get()
        ImageExtension = EntryExt.get()
        ModelPath = EntryModel.get()
        OutputDir = EntryOutputDir.get()
        ImgSize = [224, 224, 3]

        if not (TestImgDir and ImageExtension and ModelPath and OutputDir):
            messagebox.showerror('エラー', 'すべての項目を入力してください')
            return

        RunInf(TestImgDir, ImageExtension, ModelPath, ImgSize, OutputDir)
        messagebox.showinfo('完了', '処理が完了しました')

    except Exception as e:
        messagebox.showerror('エラー', str(e))

Root = tk.Tk()
Root.title('崩壊地の裸地検出（Ver. 1.0.0）')

tk.Label(Root, text='テスト画像フォルダ').grid(row=0, column=0, sticky='e')
EntryImgDir = tk.Entry(Root, width=40)
EntryImgDir.grid(row=0, column=1)
tk.Button(Root, text='参照', command=lambda: SelectDir(EntryImgDir)).grid(row=0, column=2)

tk.Label(Root, text='画像の拡張子（例:png）').grid(row=1, column=0, sticky='e')
EntryExt = tk.Entry(Root, width=20)
EntryExt.grid(row=1, column=1, sticky='w')

tk.Label(Root, text='モデルファイル').grid(row=2, column=0, sticky='e')
EntryModel = tk.Entry(Root, width=40)
EntryModel.grid(row=2, column=1)
tk.Button(Root, text='参照', command=lambda: SelectFile(EntryModel)).grid(row=2, column=2)

tk.Label(Root, text='出力フォルダ').grid(row=4, column=0, sticky='e')
EntryOutputDir = tk.Entry(Root, width=40)
EntryOutputDir.grid(row=4, column=1)
tk.Button(Root, text='参照', command=lambda: SelectDir(EntryOutputDir)).grid(row=4, column=2)

tk.Button(Root, text='実行', command=OnRun, width=20).grid(row=5, column=0, columnspan=3, pady=10)

Root.mainloop()
