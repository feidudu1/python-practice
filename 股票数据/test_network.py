"""
测试网络连接和AKShare接口
"""
import akshare as ak
import pandas as pd
import time

print("测试网络连接...")
print("=" * 60)

# 测试1: 尝试获取恒生科技指数数据（少量数据测试）
print("\n测试1: 尝试获取恒生科技指数数据（最近10条）...")
try:
    df = ak.stock_hk_index_daily_em(symbol="HSTECH")
    print(f"✓ 成功获取数据，共 {len(df)} 条记录")
    print(f"数据列名: {df.columns.tolist()}")
    print(f"最新5条数据:")
    print(df.head())
    print("\n数据日期范围:")
    if 'date' in df.columns:
        print(f"最早: {df['date'].min()}")
        print(f"最新: {df['date'].max()}")
except Exception as e:
    print(f"✗ 获取失败: {e}")
    print(f"错误类型: {type(e).__name__}")

# 测试2: 检查是否有2000-2019年的数据
print("\n" + "=" * 60)
print("测试2: 检查数据日期范围...")
try:
    df = ak.stock_hk_index_daily_em(symbol="HSTECH")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print(f"数据日期范围: {df['date'].min()} 至 {df['date'].max()}")
        
        # 检查2000-2019年的数据
        df_2000_2019 = df[(df['date'] >= '2000-01-01') & (df['date'] <= '2019-12-31')]
        print(f"2000-2019年数据: {len(df_2000_2019)} 条记录")
        
        if len(df_2000_2019) > 0:
            print(f"✓ 有2000-2019年的数据")
        else:
            print(f"✗ 没有2000-2019年的数据，可能该指数在2000年之前不存在")
except Exception as e:
    print(f"✗ 检查失败: {e}")

print("\n" + "=" * 60)
print("测试完成")

