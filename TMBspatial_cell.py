from numpy import ma, mat
import pandas as pd

#细胞类
#属性包含id与坐标
class Cell:
    snv = None
    cnv = None
    def __init__(self,id,coordinate):
        self.id = id
        self.coordinate=coordinate
        self.gene_expression = {}
        # 存储基因表达量数据，格式为{基因名: 表达量}
    def set_snv(self,snv):
        self.snv=snv
    def set_cnv(self,cnv):
        self.cnv=cnv


#组织类
#x，y是坐标细范围
class Organization:
    def __init__(self,rows,cols):
        self.x_length=rows
        self.y_length=cols
        self.gene_order = []
        # 存储基因名称的顺序，确保后续处理和输出一致性
    cell_dict = {}
    matrix = None

    def add(self,cell):
        self.cell_dict.update(cell)


    def set_matrix(self,matrix):
        self.matrix=matrix

    def save_to_csv(self,out_path):
        cell_table = []
        for v in self.cell_dict.values():
            cell_table.append([v.id,v.coordinate,v.snv,v.cnv])
        # 创建一个 DataFrame 对象
        df = pd.DataFrame(cell_table)
        # 将 DataFrame 写入 CSV 文件
        df.to_csv(out_path + '/' + 'cell.csv', index=False, header=False,encoding='utf-8')
    
    @classmethod
    def from_csv(cls, csv_path, rows=None, cols=None):
        """
        从CSV文件创建Organization对象
        :param csv_path: CSV文件路径
        :param rows: 组织x长度(可选)
        :param cols: 组织y长度(可选)
        :return: Organization对象
        """
        df = pd.read_csv(csv_path, header=None)
        org = cls(rows if rows else df.shape[0], 
                 cols if cols else df.shape[1])
        
        for _, row in df.iterrows():
            cell_id, coord, snv, cnv = row
            cell = Cell(cell_id, coord)
            if pd.notna(snv):
                cell.set_snv(snv)
            if pd.notna(cnv):
                cell.set_cnv(cnv)
            org.add({cell.id: cell})
            
        return org
        