# coding=UTF-8
import xlrd
import xlwt
from xlutils.copy import copy
# from sklearn.tree import DecisionTreeClassifier





# write_excel_xls(book_name_xls, sheet_name_xls, value_title)
# write_excel_xls_append(book_name_xls, value1)
# write_excel_xls_append(book_name_xls, value2)
# read_excel_xls(book_name_xls)

class ExcelTool:

    """
    book_name_xls = 'xls格式测试工作簿.xls'

    sheet_name_xls = 'xls格式测试表'

    value_title = [["姓名", "性别", "年龄", "城市", "职业"], ]

    value1 = [["张三", "男", "19", "杭州", "研发工程师"],
              ["李四", "男", "22", "北京", "医生"],
              ["王五", "女", "33", "珠海", "出租车司机"], ]

    value2 = [["Tom", "男", "21", "西安", "测试工程师"],
              ["Jones", "女", "34", "上海", "产品经理"],
              ["Cat", "女", "56", "上海", "教师"], ]


    """
    def __init__(self,path,book_name_xls,sheet_name_xls) :
        super(ExcelTool).__init__()
        self.path = path
        self.book_name_xls = book_name_xls
        self.excel_loc = path + book_name_xls
        self.sheet_name_xls = sheet_name_xls

    def write_xls(self, value):
        index = len(value)  # 获取需要写入数据的行数
        workbook = xlwt.Workbook()  # 新建一个工作簿
        sheet = workbook.add_sheet(self.sheet_name_xls)  # 在工作簿中新建一个表格
        for i in range(0, index):
            for j in range(0, len(value[i])):
                sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
        workbook.save(self.excel_loc)  # 保存工作簿
        print("xls add success！")

    def write_xls_append(self, value):
        index = len(value)  # 获取需要写入数据的行数
        workbook = xlrd.open_workbook(self.excel_loc)  # 打开工作簿
        sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
        worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
        rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
        new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
        new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
        for i in range(0, index):
            for j in range(0, len(value[i])):
                new_worksheet.write(i + rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
        new_workbook.save(self.excel_loc)  # 保存工作簿
        print("xls append success！")


    def write_append_blankline(self, length):
        """

        :param length:
            -------
            length: the size of add blankline
        :return:
            ------
            none
        """
        index = length  # 获取需要写入数据的行数
        workbook = xlrd.open_workbook(self.excel_loc)  # 打开工作簿
        sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
        worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
        rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
        new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
        new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
        for i in range(0, index):
            for j in range(0, 2):
                new_worksheet.write(i + rows_old, j, " ")  # 追加写入数据，注意是从i+rows_old行开始写入
        new_workbook.save(self.excel_loc)  # 保存工作簿
        print("xls append success！")

    def write_append_value_blank(self,value,length):
        self.write_xls_append(value)
        self.write_append_blankline(length)

    def read_excel_xls(self):
        workbook = xlrd.open_workbook(self.excel_loc)  # 打开工作簿
        sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
        worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
        for i in range(0, worksheet.nrows):
            for j in range(0, worksheet.ncols):
                print(worksheet.cell_value(i, j), "\t", end="")  # 逐行逐列读取数据
            print()


def test():
    ET = ExcelTool("excel/",'xls格式测试工作簿.xls','xls格式测试表')
    value_title = [["姓名", "性别", "年龄", "城市", "职业"], ]

    value1 = [["张三", "男", "19", "杭州", "研发工程师"],
              ["李四", "男", "22", "北京", "医生"],
              ["王五", "女", "33", "珠海", "出租车司机"], ]

    value2 = [["Tom", "男", "21", "西安", "测试工程师"],
              ["Jones", "女", "34", "上海", "产品经理"],
              ["Cat", "女", "56", "上海", "教师"], ]
    ET.write_excel_xls(value_title)
    ET.write_excel_xls_append(value1)
    ET.write_excel_xls_append(value2)
    ET.read_excel_xls()

# test()