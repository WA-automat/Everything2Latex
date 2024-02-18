# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lat.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_uiMainWindow(object):
    def setupUi(self, uiMainWindow):
        uiMainWindow.setObjectName("uiMainWindow")
        uiMainWindow.resize(1306, 1047)
        uiMainWindow.setMinimumSize(QtCore.QSize(591, 412))
        self.centralwidget = QtWidgets.QWidget(uiMainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setMinimumSize(QtCore.QSize(581, 412))
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setEnabled(True)
        self.widget_2.setGeometry(QtCore.QRect(10, 21, 1171, 871))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setMinimumSize(QtCore.QSize(581, 421))
        self.widget_2.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.widget_2.setAutoFillBackground(False)
        self.widget_2.setStyleSheet("\n"
"background-color: rgb(233, 238, 253);\n"
"border-radius:10px;\n"
"")
        self.widget_2.setObjectName("widget_2")
        self.stackedWidget = QtWidgets.QStackedWidget(self.widget_2)
        self.stackedWidget.setGeometry(QtCore.QRect(96, 45, 1051, 791))
        self.stackedWidget.setMinimumSize(QtCore.QSize(521, 381))
        self.stackedWidget.setObjectName("stackedWidget")
        self.pic_page = QtWidgets.QWidget()
        self.pic_page.setObjectName("pic_page")
        self.pushButton_5 = QtWidgets.QPushButton(self.pic_page)
        self.pushButton_5.setGeometry(QtCore.QRect(880, 410, 151, 111))
        self.pushButton_5.setMinimumSize(QtCore.QSize(51, 41))
        self.pushButton_5.setStyleSheet("background-color: rgb(119, 147, 246);\n"
"border-radius:10px;")
        self.pushButton_5.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/识别栏/icon/批量识别.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_5.setIcon(icon)
        self.pushButton_5.setIconSize(QtCore.QSize(48, 48))
        self.pushButton_5.setObjectName("pushButton_5")
        self.inpic = QtWidgets.QLabel(self.pic_page)
        self.inpic.setEnabled(False)
        self.inpic.setGeometry(QtCore.QRect(20, 11, 841, 291))
        self.inpic.setMinimumSize(QtCore.QSize(401, 131))
        self.inpic.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.inpic.setObjectName("inpic")
        self.outpic = QtWidgets.QLabel(self.pic_page)
        self.outpic.setEnabled(False)
        self.outpic.setGeometry(QtCore.QRect(20, 320, 841, 202))
        self.outpic.setMinimumSize(QtCore.QSize(401, 101))
        self.outpic.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.outpic.setObjectName("outpic")
        self.outthree = QtWidgets.QTextBrowser(self.pic_page)
        self.outthree.setGeometry(QtCore.QRect(20, 720, 841, 62))
        self.outthree.setMinimumSize(QtCore.QSize(401, 31))
        self.outthree.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.outthree.setObjectName("outthree")
        self.outtwo = QtWidgets.QTextBrowser(self.pic_page)
        self.outtwo.setGeometry(QtCore.QRect(20, 640, 841, 62))
        self.outtwo.setMinimumSize(QtCore.QSize(401, 31))
        self.outtwo.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.outtwo.setObjectName("outtwo")
        self.outone = QtWidgets.QTextBrowser(self.pic_page)
        self.outone.setGeometry(QtCore.QRect(20, 560, 841, 62))
        self.outone.setMinimumSize(QtCore.QSize(401, 31))
        self.outone.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.outone.setObjectName("outone")
        self.layoutWidget = QtWidgets.QWidget(self.pic_page)
        self.layoutWidget.setGeometry(QtCore.QRect(870, 550, 181, 241))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.copy1button = QtWidgets.QPushButton(self.layoutWidget)
        self.copy1button.setMinimumSize(QtCore.QSize(79, 16))
        self.copy1button.setStyleSheet("border-radius:10px;")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/文本栏/icon/复制文本.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.copy1button.setIcon(icon1)
        self.copy1button.setIconSize(QtCore.QSize(48, 48))
        self.copy1button.setObjectName("copy1button")
        self.verticalLayout.addWidget(self.copy1button)
        self.copy2button = QtWidgets.QPushButton(self.layoutWidget)
        self.copy2button.setMinimumSize(QtCore.QSize(79, 16))
        self.copy2button.setStyleSheet("border-radius:10px;")
        self.copy2button.setIcon(icon1)
        self.copy2button.setIconSize(QtCore.QSize(48, 48))
        self.copy2button.setObjectName("copy2button")
        self.verticalLayout.addWidget(self.copy2button)
        self.copy3button = QtWidgets.QPushButton(self.layoutWidget)
        self.copy3button.setMinimumSize(QtCore.QSize(79, 16))
        self.copy3button.setStyleSheet("border-radius:10px;")
        self.copy3button.setIcon(icon1)
        self.copy3button.setIconSize(QtCore.QSize(48, 48))
        self.copy3button.setObjectName("copy3button")
        self.verticalLayout.addWidget(self.copy3button)
        self.layoutWidget_2 = QtWidgets.QWidget(self.pic_page)
        self.layoutWidget_2.setGeometry(QtCore.QRect(860, 10, 191, 291))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.clear_button = QtWidgets.QPushButton(self.layoutWidget_2)
        self.clear_button.setMinimumSize(QtCore.QSize(79, 16))
        self.clear_button.setStyleSheet("border-radius:10px;")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/图片/icon/清空.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clear_button.setIcon(icon2)
        self.clear_button.setIconSize(QtCore.QSize(48, 48))
        self.clear_button.setObjectName("clear_button")
        self.verticalLayout_2.addWidget(self.clear_button)
        self.uppic = QtWidgets.QPushButton(self.layoutWidget_2)
        self.uppic.setMinimumSize(QtCore.QSize(79, 16))
        self.uppic.setStyleSheet("border-radius:10px;")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/图片/icon/截图 (1).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.uppic.setIcon(icon3)
        self.uppic.setIconSize(QtCore.QSize(48, 48))
        self.uppic.setObjectName("uppic")
        self.verticalLayout_2.addWidget(self.uppic)
        self.cutpic = QtWidgets.QPushButton(self.layoutWidget_2)
        self.cutpic.setMinimumSize(QtCore.QSize(79, 16))
        self.cutpic.setStyleSheet("border-radius:10px;")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/图片/icon/截图 (2).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.cutpic.setIcon(icon4)
        self.cutpic.setIconSize(QtCore.QSize(48, 48))
        self.cutpic.setObjectName("cutpic")
        self.verticalLayout_2.addWidget(self.cutpic)
        self.pushButton_5.raise_()
        self.outpic.raise_()
        self.outthree.raise_()
        self.outtwo.raise_()
        self.layoutWidget.raise_()
        self.layoutWidget_2.raise_()
        self.inpic.raise_()
        self.outone.raise_()
        self.stackedWidget.addWidget(self.pic_page)
        self.paint_page = QtWidgets.QWidget()
        self.paint_page.setObjectName("paint_page")
        self.outpaint = QtWidgets.QLabel(self.paint_page)
        self.outpaint.setEnabled(False)
        self.outpaint.setGeometry(QtCore.QRect(20, 320, 841, 202))
        self.outpaint.setMinimumSize(QtCore.QSize(401, 101))
        self.outpaint.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.outpaint.setObjectName("outpaint")
        self.inpic_2 = QtWidgets.QLabel(self.paint_page)
        self.inpic_2.setEnabled(False)
        self.inpic_2.setGeometry(QtCore.QRect(20, 11, 841, 291))
        self.inpic_2.setMinimumSize(QtCore.QSize(401, 131))
        self.inpic_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.inpic_2.setObjectName("inpic_2")
        self.outone_2 = QtWidgets.QTextBrowser(self.paint_page)
        self.outone_2.setGeometry(QtCore.QRect(20, 560, 841, 62))
        self.outone_2.setMinimumSize(QtCore.QSize(401, 31))
        self.outone_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.outone_2.setObjectName("outone_2")
        self.outthree_2 = QtWidgets.QTextBrowser(self.paint_page)
        self.outthree_2.setGeometry(QtCore.QRect(20, 720, 841, 62))
        self.outthree_2.setMinimumSize(QtCore.QSize(401, 31))
        self.outthree_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.outthree_2.setObjectName("outthree_2")
        self.outtwo_2 = QtWidgets.QTextBrowser(self.paint_page)
        self.outtwo_2.setGeometry(QtCore.QRect(20, 640, 841, 62))
        self.outtwo_2.setMinimumSize(QtCore.QSize(401, 31))
        self.outtwo_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.outtwo_2.setObjectName("outtwo_2")
        self.layoutWidget_3 = QtWidgets.QWidget(self.paint_page)
        self.layoutWidget_3.setGeometry(QtCore.QRect(860, 10, 191, 291))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.clearpaint_butt = QtWidgets.QPushButton(self.layoutWidget_3)
        self.clearpaint_butt.setMinimumSize(QtCore.QSize(59, 16))
        self.clearpaint_butt.setStyleSheet("border-radius:10px;")
        self.clearpaint_butt.setIcon(icon2)
        self.clearpaint_butt.setIconSize(QtCore.QSize(48, 48))
        self.clearpaint_butt.setObjectName("clearpaint_butt")
        self.verticalLayout_3.addWidget(self.clearpaint_butt)
        self.pan = QtWidgets.QPushButton(self.layoutWidget_3)
        self.pan.setMinimumSize(QtCore.QSize(59, 16))
        self.pan.setStyleSheet("border-radius:10px;")
        self.pan.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/手写/icon/画笔 (1).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pan.setIcon(icon5)
        self.pan.setIconSize(QtCore.QSize(48, 48))
        self.pan.setObjectName("pan")
        self.verticalLayout_3.addWidget(self.pan)
        self.changepan = QtWidgets.QPushButton(self.layoutWidget_3)
        self.changepan.setMinimumSize(QtCore.QSize(59, 16))
        self.changepan.setStyleSheet("border-radius:10px;")
        self.changepan.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/手写/icon/圆点.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.changepan.setIcon(icon6)
        self.changepan.setIconSize(QtCore.QSize(48, 48))
        self.changepan.setObjectName("changepan")
        self.verticalLayout_3.addWidget(self.changepan)
        self.back = QtWidgets.QPushButton(self.layoutWidget_3)
        self.back.setMinimumSize(QtCore.QSize(59, 16))
        self.back.setStyleSheet("border-radius:10px;")
        self.back.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/手写/icon/撤回.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.back.setIcon(icon7)
        self.back.setIconSize(QtCore.QSize(48, 48))
        self.back.setObjectName("back")
        self.verticalLayout_3.addWidget(self.back)
        self.layoutWidget_4 = QtWidgets.QWidget(self.paint_page)
        self.layoutWidget_4.setGeometry(QtCore.QRect(870, 550, 181, 241))
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.layoutWidget_4)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.copy1button_2 = QtWidgets.QPushButton(self.layoutWidget_4)
        self.copy1button_2.setMinimumSize(QtCore.QSize(79, 16))
        self.copy1button_2.setStyleSheet("border-radius:10px;")
        self.copy1button_2.setIcon(icon1)
        self.copy1button_2.setIconSize(QtCore.QSize(48, 48))
        self.copy1button_2.setObjectName("copy1button_2")
        self.verticalLayout_4.addWidget(self.copy1button_2)
        self.copy2button_2 = QtWidgets.QPushButton(self.layoutWidget_4)
        self.copy2button_2.setMinimumSize(QtCore.QSize(79, 16))
        self.copy2button_2.setStyleSheet("border-radius:10px;")
        self.copy2button_2.setIcon(icon1)
        self.copy2button_2.setIconSize(QtCore.QSize(48, 48))
        self.copy2button_2.setObjectName("copy2button_2")
        self.verticalLayout_4.addWidget(self.copy2button_2)
        self.copy3button_2 = QtWidgets.QPushButton(self.layoutWidget_4)
        self.copy3button_2.setMinimumSize(QtCore.QSize(79, 16))
        self.copy3button_2.setStyleSheet("border-radius:10px;")
        self.copy3button_2.setIcon(icon1)
        self.copy3button_2.setIconSize(QtCore.QSize(48, 48))
        self.copy3button_2.setObjectName("copy3button_2")
        self.verticalLayout_4.addWidget(self.copy3button_2)
        self.pushButton_6 = QtWidgets.QPushButton(self.paint_page)
        self.pushButton_6.setGeometry(QtCore.QRect(880, 410, 151, 111))
        self.pushButton_6.setMinimumSize(QtCore.QSize(51, 41))
        self.pushButton_6.setStyleSheet("background-color: rgb(119, 147, 246);\n"
"border-radius:10px;")
        self.pushButton_6.setText("")
        self.pushButton_6.setIcon(icon)
        self.pushButton_6.setIconSize(QtCore.QSize(48, 48))
        self.pushButton_6.setObjectName("pushButton_6")
        self.stackedWidget.addWidget(self.paint_page)
        self.about_page = QtWidgets.QWidget()
        self.about_page.setObjectName("about_page")
        self.stackedWidget.addWidget(self.about_page)
        self.EXIT = QtWidgets.QPushButton(self.widget_2)
        self.EXIT.setGeometry(QtCore.QRect(980, 10, 141, 31))
        self.EXIT.setMinimumSize(QtCore.QSize(31, 21))
        self.EXIT.setStyleSheet("border-radius:10px;\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(119, 147, 246);\n"
"font: 24pt \"Agency FB\";")
        self.EXIT.setIconSize(QtCore.QSize(48, 48))
        self.EXIT.setObjectName("EXIT")
        self.lead = QtWidgets.QFrame(self.widget_2)
        self.lead.setGeometry(QtCore.QRect(0, 0, 91, 871))
        self.lead.setMinimumSize(QtCore.QSize(51, 421))
        self.lead.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0.451599, y1:0.755955, x2:0.440484, y2:0.357818, stop:0 rgba(255, 255, 255, 247), stop:1 rgba(112, 142, 247, 241));\n"
"border-radius:10px;")
        self.lead.setObjectName("lead")
        self.about_buttpn = QtWidgets.QPushButton(self.lead)
        self.about_buttpn.setGeometry(QtCore.QRect(20, 680, 59, 50))
        self.about_buttpn.setMinimumSize(QtCore.QSize(31, 41))
        self.about_buttpn.setStyleSheet("background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 0, 0, 0), stop:1 rgba(255, 255, 255, 0));\n"
"border:none;")
        self.about_buttpn.setText("")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/导航栏/icon/关于我们.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.about_buttpn.setIcon(icon8)
        self.about_buttpn.setIconSize(QtCore.QSize(40, 40))
        self.about_buttpn.setObjectName("about_buttpn")
        self.pic_button = QtWidgets.QPushButton(self.lead)
        self.pic_button.setGeometry(QtCore.QRect(21, 138, 59, 50))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pic_button.sizePolicy().hasHeightForWidth())
        self.pic_button.setSizePolicy(sizePolicy)
        self.pic_button.setMinimumSize(QtCore.QSize(31, 41))
        self.pic_button.setTabletTracking(False)
        self.pic_button.setToolTipDuration(-1)
        self.pic_button.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pic_button.setAutoFillBackground(False)
        self.pic_button.setStyleSheet("background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 0, 0, 0), stop:1 rgba(255, 255, 255, 0));\n"
"border:none;")
        self.pic_button.setText("")
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/导航栏/icon/截图.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pic_button.setIcon(icon9)
        self.pic_button.setIconSize(QtCore.QSize(40, 40))
        self.pic_button.setCheckable(False)
        self.pic_button.setAutoRepeat(False)
        self.pic_button.setAutoExclusive(False)
        self.pic_button.setAutoRepeatDelay(300)
        self.pic_button.setAutoDefault(False)
        self.pic_button.setObjectName("pic_button")
        self.paint_button = QtWidgets.QPushButton(self.lead)
        self.paint_button.setGeometry(QtCore.QRect(21, 271, 59, 50))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.paint_button.sizePolicy().hasHeightForWidth())
        self.paint_button.setSizePolicy(sizePolicy)
        self.paint_button.setMinimumSize(QtCore.QSize(31, 41))
        self.paint_button.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.paint_button.setStyleSheet("background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 0, 0, 0), stop:1 rgba(255, 255, 255, 0));\n"
"border:none;")
        self.paint_button.setText("")
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/导航栏/icon/画笔.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.paint_button.setIcon(icon10)
        self.paint_button.setIconSize(QtCore.QSize(40, 40))
        self.paint_button.setObjectName("paint_button")
        uiMainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(uiMainWindow)
        self.stackedWidget.setCurrentIndex(0)
        self.clearpaint_butt.clicked.connect(self.inpic_2.clear) # type: ignore
        self.copy1button_2.clicked.connect(self.outone_2.copy) # type: ignore
        self.copy2button_2.clicked.connect(self.outtwo_2.copy) # type: ignore
        self.copy3button_2.clicked.connect(self.outthree_2.copy) # type: ignore
        self.pushButton_6.clicked.connect(self.outpaint.show) # type: ignore
        self.EXIT.clicked.connect(uiMainWindow.close) # type: ignore
        self.clear_button.clicked.connect(self.inpic.clear) # type: ignore
        self.copy1button.clicked.connect(self.outone.copy) # type: ignore
        self.copy2button.clicked.connect(self.outtwo.copy) # type: ignore
        self.copy3button.clicked.connect(self.outthree.copy) # type: ignore
        self.pushButton_5.clicked.connect(self.outpic.show) # type: ignore
        self.pic_button.clicked.connect(uiMainWindow.turntopage_pic) # type: ignore
        self.paint_button.clicked.connect(uiMainWindow.turntopage_paint) # type: ignore
        self.about_buttpn.clicked.connect(uiMainWindow.turntopage_about) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(uiMainWindow)

    def retranslateUi(self, uiMainWindow):
        _translate = QtCore.QCoreApplication.translate
        uiMainWindow.setWindowTitle(_translate("uiMainWindow", "MainWindow"))
        self.inpic.setText(_translate("uiMainWindow", "上传图片或截图"))
        self.outpic.setText(_translate("uiMainWindow", "识别结果"))
        self.copy1button.setText(_translate("uiMainWindow", "copy"))
        self.copy2button.setText(_translate("uiMainWindow", "copy"))
        self.copy3button.setText(_translate("uiMainWindow", "copy"))
        self.clear_button.setText(_translate("uiMainWindow", "清空图片"))
        self.uppic.setText(_translate("uiMainWindow", "上传图片"))
        self.cutpic.setText(_translate("uiMainWindow", "截取图片"))
        self.outpaint.setText(_translate("uiMainWindow", "识别结果"))
        self.inpic_2.setText(_translate("uiMainWindow", "画板"))
        self.clearpaint_butt.setText(_translate("uiMainWindow", "清空"))
        self.copy1button_2.setText(_translate("uiMainWindow", "copy"))
        self.copy2button_2.setText(_translate("uiMainWindow", "copy"))
        self.copy3button_2.setText(_translate("uiMainWindow", "copy"))
        self.EXIT.setText(_translate("uiMainWindow", "X"))
        self.about_buttpn.setWhatsThis(_translate("uiMainWindow", "<html><head/><body><p>关于</p></body></html>"))
        self.paint_button.setWhatsThis(_translate("uiMainWindow", "<html><head/><body><p>手写识别</p></body></html>"))
import resource_rc