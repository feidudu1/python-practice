"""
修复恒生科技指数数据 - 使用备用方法获取数据
"""

import akshare as ak
import pandas as pd


def get_hstech_data_alternative():
    """
    使用备用方法获取恒生科技指数数据
    """
    print("正在尝试备用方法获取数据...")

    # 尝试使用 tool_trade_date_hist_sina 获取交易日列表
    # 然后尝试其他接口

    # 方法：尝试直接使用 stock_hk_index_daily_em 但使用不同的参数
    try:
        print("尝试方法: 直接调用接口...")
        # 先获取指数列表，找到HSTECH的代码
        # 然后获取历史数据

        # 尝试使用 tool_trade_date_hist_sina 接口
        print("获取交易日列表...")
        trade_dates = ak.tool_trade_date_hist_sina()
        print(f"获取到 {len(trade_dates)} 个交易日")

        # 尝试使用其他方式获取恒生科技指数
        # 恒生科技指数的代码可能是 "HSTECH" 或 "HSTECH.HI"
        print("尝试获取恒生科技指数数据...")

        # 使用 stock_hk_index_daily_em 接口，但可能需要不同的参数
        # 先尝试获取所有港股指数列表
        try:
            index_list = ak.stock_hk_index_spot_em()
            print("可用的港股指数:")
            print(index_list.head(10))

            # 查找恒生科技指数
            hstech_info = index_list[
                index_list["指数名称"].str.contains("科技", na=False)
            ]
            print("\n包含'科技'的指数:")
            print(hstech_info)

        except Exception as e:
            print(f"获取指数列表失败: {e}")

        # 直接尝试获取数据
        try:
            df = ak.stock_hk_index_daily_em(symbol="HSTECH")
            print(f"成功获取数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"直接获取失败: {e}")

    except Exception as e:
        print(f"备用方法失败: {e}")

    return None


def process_and_save_data():
    """
    处理并保存数据
    """
    df = get_hstech_data_alternative()

    if df is None or df.empty:
        print("无法获取数据")
        return

    print("\n原始数据列名:")
    print(df.columns.tolist())
    print("\n原始数据前5行:")
    print(df.head())

    # 处理数据
    result_df = pd.DataFrame()

    # 处理日期列
    date_col = None
    for col in ["日期", "date", "Date", "时间"]:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        result_df["日期"] = pd.to_datetime(df[date_col])
    else:
        try:
            result_df["日期"] = pd.to_datetime(df.iloc[:, 0])
        except:
            print("无法识别日期列")
            return None

    # 处理价格列
    col_mapping = {
        "开盘价": ["开盘", "open", "Open", "OPEN"],
        "收盘价": ["收盘", "close", "Close", "CLOSE", "latest", "Latest", "LATEST"],
        "最高价": ["最高", "high", "High", "HIGH"],
        "最低价": ["最低", "low", "Low", "LOW"],
    }

    for target_col, possible_cols in col_mapping.items():
        found = False
        for source_col in possible_cols:
            if source_col in df.columns:
                result_df[target_col] = pd.to_numeric(df[source_col], errors="coerce")
                found = True
                break

        if not found:
            print(f"警告: 未找到 {target_col} 对应的列")
            result_df[target_col] = None

    # 按日期排序并筛选
    result_df = result_df.sort_values("日期").reset_index(drop=True)
    result_df = result_df[
        (result_df["日期"] >= "2020-01-01") & (result_df["日期"] <= "2025-12-31")
    ]

    # 计算涨跌幅
    if "收盘价" in result_df.columns and result_df["收盘价"].notna().any():
        result_df["单日涨跌幅"] = result_df["收盘价"].pct_change() * 100
    else:
        result_df["单日涨跌幅"] = None

    # 格式化数据
    formatted_df = pd.DataFrame()

    # 日期格式
    formatted_df["日期"] = result_df["日期"].apply(
        lambda x: f"{x.year}/{x.month}/{x.day}" if pd.notna(x) else ""
    )

    # 价格格式
    def format_price(value):
        if pd.isna(value):
            return ""
        return f"{value:,.2f}"

    formatted_df["收盘"] = (
        result_df["收盘价"].apply(format_price) if "收盘价" in result_df.columns else ""
    )
    formatted_df["开盘"] = (
        result_df["开盘价"].apply(format_price) if "开盘价" in result_df.columns else ""
    )
    formatted_df["高"] = (
        result_df["最高价"].apply(format_price) if "最高价" in result_df.columns else ""
    )
    formatted_df["低"] = (
        result_df["最低价"].apply(format_price) if "最低价" in result_df.columns else ""
    )
    formatted_df["交易量"] = ""  # 指数数据通常没有交易量
    formatted_df["涨跌幅"] = (
        result_df["单日涨跌幅"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
        if "单日涨跌幅" in result_df.columns
        else ""
    )

    # 保存
    output_file = "恒生科技指数_2020-2025.csv"
    formatted_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n数据已保存至: {output_file}")
    print(f"共 {len(formatted_df)} 条记录")
    print("\n数据预览（前10行）:")
    print(formatted_df.head(10))


if __name__ == "__main__":
    process_and_save_data()
