import rootutils
rootutils.setup_root(__file__,indicator='.project-root',pythonpath=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim

import math
import os
import sys
import csv
import math
from functools import reduce
from collections import Counter

import numpy as np
import pandas as pd

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QVBoxLayout,QSplitter,QWidget,QLabel,QTextEdit,QPushButton,QTableWidgetItem,
                              QMainWindow,QDesktopWidget,QApplication,QGridLayout,QTableWidget,QAbstractItemView,
                                QMessageBox,qApp,QFileDialog,QRadioButton,QHBoxLayout)
from PyQt5.QtCore import Qt,QObject,pyqtSignal

import qdarkgraystyle

from src.utils.utils import is_number
from src.predict import predict_main
from src.constants import SITE_SPECIES, SPECIES

WINDOES_TITLE='XXXXXX'

MODEL_PATH={'Basic':'../models/basic.pkl','C. elegans':'../models/C.pkl','D. melanogaster':'../models/D.pkl','A. thaliana':'../models/A.pkl','E. coli':'../models/E.pkl','G. subterraneus':'../models/Gsub20.pkl','G. pickeringii':'../models/Gpick5.pkl'}

class fileBtnWidget(QWidget,QObject):
    input_signal=pyqtSignal(object)
    input_state_signal=pyqtSignal(object)
    """File Widget"""
    def __init__(self,name):
        super().__init__()
        self.name=name
        self.res=None
        self.initUI()

    def initUI(self):
        if self.name=='import':
            importBtn = QPushButton("Import", self)
            importBtn.clicked.connect(self.importButtonClicked)
            fileGrid = QGridLayout()
            fileGrid.addWidget(importBtn,0,0)
            self.setLayout(fileGrid)
        if self.name=='export':
            exportBtn = QPushButton("Export", self)
            exportBtn.clicked.connect(self.exportButtonClicked)
            fileGrid = QGridLayout()
            fileGrid.addWidget(exportBtn,0,0)
            self.setLayout(fileGrid)

    def clear_res(self):
        self.res=None

    def toMB(self,bytesize):
        return f'{bytesize/1024/1024:.2f}'
      
    def importButtonClicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file','','Text files (*.csv)')

        if fname[0]:
            self.input_state_signal.emit('Loading data......')
            data=pd.read_csv(fname[0])
            self.input_signal.emit(data)
            self.input_state_signal.emit('File loaded successfully')
      
    def exportButtonClicked(self):
        if self.res is not None:
            fname,ok= QFileDialog.getSaveFileName(self, 'Save file','','csv(*.csv)')
            if fname:
                self.res.to_csv(fname,index=False)
                    
                QMessageBox.about(self,'Success',"File saved successfully.")
        else:
            QMessageBox.about(self,'Error',"The result is empty.")
    
    def refresh_result_data(self,data):
        self.res=data

class PdtBtnWidget(QWidget,QObject):
    '''Predict functional related widgets'''

    predict_signal=pyqtSignal(object)
    input_signal=pyqtSignal(object)
    clear_signal=pyqtSignal()

    def __init__(self):
        super().__init__()
        self.input_data=None
        self.type = 'site'
        self.site = '4mC'
        self.species = ''
        
        self.initUI()

    def initUI(self):
        exampleBtn = QPushButton("Data format example", self)
        exampleBtn.clicked.connect(self.examplebuttonClicked)

        predictBtn = QPushButton("Predict", self)
        predictBtn.clicked.connect(self.prebuttonClicked)

        clearBtn = QPushButton("Clear", self)
        clearBtn.clicked.connect(self.clearbuttonClicked)

        exitBtn = QPushButton("Exit", self)
        exitBtn.clicked.connect(self.exitClicked)

        pdtGrid = QGridLayout()
        pdtGrid.addWidget(predictBtn,0,0)
        pdtGrid.addWidget(exampleBtn,0,1)
        pdtGrid.addWidget(clearBtn,1,0)
        pdtGrid.addWidget(exitBtn,1,1)
        self.setLayout(pdtGrid)

    def refresh_input_data(self,input_data):
        self.input_data=input_data

    def refresh_model_path(self,model_name):
        split = model_name.split('_')
        
        if len(split) == 1:
            self.type = 'site'
            self.site = split[0]
            self.species = ''
        else:
            self.type = 'site_species'
            self.site = split[0]
            self.species = split[1]
        

    def prebuttonClicked(self):
        if self.input_data is not None:
            # if len(self.input_data) > 1000:
            # QMessageBox.about(self,'Info','Please wait patiently for the result.')
            res, msg=predict_main(self.input_data,self.type, self.site, self.species)
            if msg =='sucess':
                self.predict_signal.emit(res)
            else:
                QMessageBox.about(self,'Error',msg)
        else:
            QMessageBox.about(self,'Error','Please input your data.')
    
    # #清除按钮操作
    def clearbuttonClicked(self):
        self.input_data=None
        self.clear_signal.emit()

    #输入案例操作
    def examplebuttonClicked(self):
        text="""
        You can click import to upload data。
        

        The CSV file to be imported must contain the columns seq and species. 
        Below are the species categories currently covered at each sit
        
        4mC:
        C equisetifolia
        F vesca
        S cerevisiae
        Tolypocladium
        
        5hmC:
        M musculus
        H sapiens
        
        6mA:
        A thaliana
        C elegans
        C equisetifolia
        D melanogast
        F vesca
        H sapiens
        R chinensis
        S cerevisiae
        T thermophil
        Tolypocladium
        Xoc
        BLS25
        """
        QMessageBox.about(self,'Data format example',text)
        data=pd.read_csv('data/4mC_demo.csv')
        self.input_signal.emit(data)

    def exitClicked(self,event):
        reply = QMessageBox.question(self, 'Message',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            # event.ignore()
            pass

class ModelBtnWidget(QWidget,QObject):
    """Model selection radio button"""

    check_signal = pyqtSignal(object)

    def __init__(self,model_name,check,font_size=14, name_prefix= ''):
        super().__init__()
        self.model_name=model_name
        self.check=check
        self.font_size=font_size
        self.name_prefix = name_prefix
        self.initUI()

    def initUI(self):
        layout=QHBoxLayout(self)
        meanModelTitle = QLabel(self.model_name)
        meanModelTitle.setFont(QFont('Arial', self.font_size))
        # meanModelTitle.setAlignment(Qt.AlignLeft)
        self.meanModelBtn=QRadioButton()
        self.meanModelBtn.setChecked(self.check)
        self.meanModelBtn.clicked.connect(lambda:self.setModelClicked(self.model_name if self.name_prefix =='' else self.name_prefix+'_'+self.model_name))

        layout.addWidget(self.meanModelBtn)
        layout.addWidget(meanModelTitle)

    def getBtnName(self):
        return self.model_name if self.name_prefix =='' else self.name_prefix+'_'+self.model_name 

    def setModelClicked(self,btnname):
        self.check_signal.emit(btnname)
        self.meanModelBtn.setChecked(True)

class MdlWidget(QWidget,QObject):
    """Model Selection Widget"""

    model_select_signal=pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.initUI()
    def initUI(self):

        self.model_list=[]

        layout=QGridLayout()

        basicTitle = QLabel('Site Model')
        basicTitle.setFont(QFont('Arial', 17))

        spTitle = QLabel('Specie Enhance Model')
        spTitle.setFont(QFont('Arial', 17))
        
        self.m0_0=ModelBtnWidget('4mC',True)
        self.model_list.append(self.m0_0)
        self.m0_0.check_signal.connect(self.buttonClicked)
        
        self.m0_1=ModelBtnWidget('5hmC',False)
        self.model_list.append(self.m0_1)
        self.m0_1.check_signal.connect(self.buttonClicked)
        
        self.m0_2=ModelBtnWidget('6mA',False)
        self.model_list.append(self.m0_2)
        self.m0_2.check_signal.connect(self.buttonClicked)
        
        layout.addWidget(basicTitle,0,0,1,3)
        layout.addWidget(self.m0_0,1,0,1,1)
        layout.addWidget(self.m0_1,1,1,1,1)
        layout.addWidget(self.m0_2,1,2,1,1)
        
        layout.addWidget(spTitle,2,0,1,1)
        cur_row = 3
        cur_model_idx = 1
        col_thr = 4
        for site in ['4mC', '5hmC', '6mA']:
            species_list = SITE_SPECIES[site]
            
            sub_spTitle = QLabel(site)
            sub_spTitle.setFont(QFont('Arial', 14))
            sub_spTitle.setAlignment(Qt.AlignCenter)
            setattr(self, f'{site}_sub_spTitle', sub_spTitle)
            sub_spTitle_widget = getattr(self, f'{site}_sub_spTitle')
            layout.addWidget(sub_spTitle_widget,cur_row,0,1,1)
            
            for index, species in enumerate(species_list):
                model_btn_widget = ModelBtnWidget(species, False, name_prefix=site)
                self.model_list.append(model_btn_widget)
                model_btn_widget.check_signal.connect(self.buttonClicked)
                attr_name = f'm{index + cur_model_idx}'
                setattr(self, attr_name, model_btn_widget)
                m_widget = getattr(self, attr_name)
                layout.addWidget(m_widget,cur_row+int(index/col_thr),index%col_thr+1,1,1)
            
            cur_model_idx = cur_model_idx + len(species_list)
            cur_row = cur_row +math.ceil(len(species_list)/col_thr)

        self.setLayout(layout)

    def buttonClicked(self,btnname):
        for i,mbtn in enumerate(self.model_list):
            if mbtn.getBtnName()!=btnname:
                self.model_list[i].meanModelBtn.setChecked(False)
        self.model_select_signal.emit(btnname)

class RightFuncBtnWidget(QWidget):
    """Fucntion Widget"""

    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.pdtBtnWidget=PdtBtnWidget()
        self.mdlWidget=MdlWidget()

        # Set the signal
        self.mdlWidget.model_select_signal.connect(self.pdtBtnWidget.refresh_model_path)
        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.mdlWidget,0,0,2,1)
        grid.addWidget(self.pdtBtnWidget,2,0,1,1)

        self.setLayout(grid)

class MyTable(QTableWidget):
    """Table Widget"""

    def __init__(self,parent=None):
        super(MyTable,self).__init__(parent)
        # set data readOnly
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # update data
        self.updateData([],[],np.array([]))
    
    # set header
    def setTableHeader(self,header):
        if header is not None:
            self.setHorizontalHeaderLabels(list(header))
		
    # set row name
    def setTableRowName(self,row_name=None):
        if row_name is not None:
            row_name=[str(i) for i in row_name]
            self.setVerticalHeaderLabels(row_name)
    
    def removeBefore(self):
        #initialize header
        self.setColumnCount(0)
        #set header
        self.setTableHeader('')
        
        rowcount = self.rowCount()
        while rowcount>0:
            rowcount = self.rowCount()
            self.removeRow(rowcount-1)

    def updateData(self,header,array,row_name=None):
        self.removeBefore()
        if array is not None and len(array)>0:
            
            # set column count
            max_columns_len=np.max([len(i) for i in array])
            max_header_len=0
            if header is not None:
                max_header_len=len(header)
            self.setColumnCount(max(max_header_len,max_columns_len))

            # set header
            self.setTableHeader(header)

            # set array
            for i in range(len(array)):
                rowcount = self.rowCount()
                self.insertRow(rowcount)
                self.cur_line=array[i]
									
                for j in range(len(self.cur_line)):
                    if is_number(self.cur_line[j]) and isinstance(self.cur_line[j],str) and self.cur_line[j].isdigit() is False:
                        self.setItem(i,j,QTableWidgetItem('%.4f'%float(self.cur_line[j])))
                    else:
                        self.setItem(i,j,QTableWidgetItem(str(self.cur_line[j])))
            
            self.resizeColumnsToContents()
            self.horizontalHeader().setStretchLastSection(True)

            # set row count and name
            self.setRowCount(len(array))
            self.setTableRowName(row_name)
            
class InputWidget(QWidget):
    """Upload files in this widget"""

    def __init__(self):
        super().__init__()
        self.init_UI()

    def init_UI(self):

        grid=QGridLayout()

        dataTitle = QLabel('Please upload your data')
        dataTitle.setFont(QFont('Arial', 20))
        self.dataTable=MyTable()
        self.importBtn=fileBtnWidget('import')

        # Set the signal
        self.importBtn.input_signal.connect(self.refreshTable)

        grid.addWidget(dataTitle,0,0,1,1)
        grid.addWidget(self.importBtn,0,1,1,1)
        grid.addWidget(self.dataTable,1,0,3,2)

        self.setLayout(grid)

    def refreshTable(self,data):
        self.dataTable.updateData(data.columns,data.values)

class ResultWidget(QWidget):
    """Displays the results and provides an export interface"""

    def __init__(self):
        super().__init__()
        self.init_UI()
    
    def init_UI(self):
        reTitle = QLabel('Prediction result')
        reTitle.setFont(QFont('Arial', 20))
        self.exportBtn=fileBtnWidget('export')
        self.reTable=MyTable()

        grid=QGridLayout(self)
        grid.addWidget(reTitle, 0, 0,1,3)
        grid.addWidget(self.reTable,1,0,3,3)
        grid.addWidget(self.exportBtn, 4, 0,1,3)
    
    def refresh_result_data(self,data):
        self.reTable.updateData(data.columns,data.values)

class ContentWidget(QWidget):
    """Main Widget contains subwidgets"""
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):        
        
        # SubWidget
        self.inputWidget=InputWidget()
        self.rfBtn=RightFuncBtnWidget()
        self.resultWidget=ResultWidget()

        # Bind the signal
        self.inputWidget.importBtn.input_signal.connect(self.rfBtn.pdtBtnWidget.refresh_input_data)
        self.rfBtn.pdtBtnWidget.input_signal.connect(self.rfBtn.pdtBtnWidget.refresh_input_data)
        self.rfBtn.pdtBtnWidget.input_signal.connect(self.inputWidget.refreshTable)
        self.rfBtn.pdtBtnWidget.predict_signal.connect(self.resultWidget.refresh_result_data)
        self.rfBtn.pdtBtnWidget.predict_signal.connect(self.resultWidget.exportBtn.refresh_result_data)
        self.rfBtn.pdtBtnWidget.clear_signal.connect(self.inputWidget.dataTable.removeBefore)
        self.rfBtn.pdtBtnWidget.clear_signal.connect(self.resultWidget.reTable.removeBefore)
        self.rfBtn.pdtBtnWidget.clear_signal.connect(self.resultWidget.exportBtn.clear_res)
        # Set the layout
        layout = QVBoxLayout()

        splitter_up = QSplitter(Qt.Horizontal)
        splitter_up.addWidget(self.inputWidget)
        splitter_up.addWidget(self.rfBtn)

        splitter= QSplitter(Qt.Vertical)
        splitter.addWidget(splitter_up)
        splitter.addWidget(self.resultWidget)
        
        layout.addWidget(splitter)

        self.setLayout(layout)
        self.setGeometry(300, 300, 350, 300)
        self.show() 

class MainWindow(QMainWindow):
    """Main Window"""
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle(WINDOES_TITLE)
        self.center()
        self.setStyleSheet(qdarkgraystyle.load_stylesheet_pyqt5())
        
        self.cw = ContentWidget()
        self.setCentralWidget(self.cw)
        self.statusBar().showMessage('')

        # Set the state
        # self.cw.inputWidget.importBtn.input_state_signal.connect(self.refresh_statusBar)
        self.show()

    def center(self):
        """Center the window"""
        self.resize(1400,950)
        self.setWindowState(Qt.WindowMaximized)
        qr=self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def refresh_statusBar(self,state):
        self.statusBar().showMessage(state)

if __name__=="__main__":
    os.chdir(sys.path[0])
    app=QApplication(sys.argv)
    window=MainWindow()
    sys.exit(app.exec_())

