"""
修复CSV文件：重新计算涨跌幅，将交易量空值改为0
"""
import pandas as pd

def fix_csv_file():
    """
    修复CSV文件
    """
    input_file = "恒生科技指数_2020-2025.csv"
    
    print(f"正在读取文件: {input_file}")
    # 读取CSV文件
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"原始数据共 {len(df)} 条记录")
    print("\n原始数据前5行:")
    print(df.head())
    
    # 将收盘价从字符串转换为数值（去掉千分位逗号）
    df['收盘_数值'] = df['收盘'].str.replace(',', '').astype(float)
    
    # 重新计算涨跌幅：((当前收盘价 - 前一天收盘价) / 前一天收盘价) × 100%
    df['涨跌幅_新'] = (
        (df['收盘_数值'] - df['收盘_数值'].shift(1)) 
        / df['收盘_数值'].shift(1) 
        * 100
    )
    
    # 格式化涨跌幅为百分比格式
    df['涨跌幅'] = df['涨跌幅_新'].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else ""
    )
    
    # 将交易量空值改为0
    df['交易量'] = df['交易量'].fillna('0')
    df['交易量'] = df['交易量'].replace('', '0')
    
    # 删除临时列
    df = df.drop(['收盘_数值', '涨跌幅_新'], axis=1)
    
    # 保存修复后的文件
    output_file = "恒生科技指数_2020-2025.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n数据已修复并保存至: {output_file}")
    print(f"共 {len(df)} 条记录")
    print("\n修复后的数据预览（前10行）:")
    print(df.head(10))
    print("\n修复后的数据预览（后10行）:")
    print(df.tail(10))
    
    # 验证涨跌幅计算
    print("\n涨跌幅计算验证（前5行）:")
    print(df[['日期', '收盘', '涨跌幅']].head(5))

if __name__ == "__main__":
    fix_csv_file()

