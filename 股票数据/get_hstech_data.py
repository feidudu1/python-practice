"""
获取恒生科技指数从2020年到2025年的每日数据
包含：日期、收盘价、开盘价、最高价、最低价、交易量、单日涨跌幅
"""

import akshare as ak
import pandas as pd


def get_hstech_data():
    """
    获取恒生科技指数历史数据
    """
    print("正在获取恒生科技指数数据...")

    try:
        # 方法1: 尝试使用 index_hk_hist 接口
        print("尝试方法1: index_hk_hist...")
        try:
            df = ak.index_hk_hist(
                symbol="HSTECH",
                period="daily",
                start_date="2000-01-01",
                end_date="2019-12-31",
            )
            print(f"成功获取数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"方法1失败: {e}")

        # 方法2: 尝试使用 stock_hk_index_daily_em 接口
        print("尝试方法2: stock_hk_index_daily_em...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = ak.stock_hk_index_daily_em(symbol="HSTECH")
                # 筛选日期范围
                if "日期" in df.columns:
                    df["日期"] = pd.to_datetime(df["日期"])
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                df = df[
                    (df.iloc[:, 0] >= "2020-01-01") & (df.iloc[:, 0] <= "2025-12-31")
                ]
                print(f"成功获取数据，共 {len(df)} 条记录")
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"方法2尝试 {attempt + 1}/{max_retries} 失败，正在重试...")
                    import time

                    time.sleep(2)  # 等待2秒后重试
                else:
                    print(f"方法2失败: {e}")

        # 方法3: 尝试使用 tool_trade_date_hist_sina 获取交易日，然后获取指数数据
        print("尝试方法3: 使用其他接口...")
        try:
            # 尝试使用 index_hk_daily_em 接口
            df = ak.index_hk_daily_em(symbol="HSTECH")
            if "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"])
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            df = df[(df.iloc[:, 0] >= "2000-01-01") & (df.iloc[:, 0] <= "2019-12-31")]
            print(f"成功获取数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"方法3失败: {e}")

        # 方法4: 尝试使用 stock_hk_index_spot_em 获取实时数据（作为备选）
        print("尝试方法4: 使用其他备选接口...")
        try:
            # 如果以上方法都失败，可以尝试分年获取
            print("注意: 将尝试分年获取数据...")
            all_data = []
            for year in range(2000, 2020):
                try:
                    year_df = ak.stock_hk_index_daily_em(symbol="HSTECH")
                    if "date" in year_df.columns:
                        year_df["date"] = pd.to_datetime(year_df["date"])
                        year_df = year_df[year_df["date"].dt.year == year]
                        if not year_df.empty:
                            all_data.append(year_df)
                            print(f"成功获取 {year} 年数据，共 {len(year_df)} 条记录")
                except Exception as e:
                    print(f"获取 {year} 年数据失败: {e}")

            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                df = df.sort_values("date").reset_index(drop=True)
                print(f"成功获取数据，共 {len(df)} 条记录")
                return df
        except Exception as e:
            print(f"方法4失败: {e}")

    except Exception as e:
        print(f"获取数据时出错: {e}")
        return None

    return None


def process_data(df):
    """
    处理数据，选择需要的列并计算涨跌幅
    """
    if df is None or df.empty:
        print("数据为空，无法处理")
        return None

    print("\n原始数据列名:")
    print(df.columns.tolist())
    print("\n原始数据前5行:")
    print(df.head())

    # 创建新的DataFrame
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
        # 如果第一列是日期类型，直接使用
        if len(df.columns) > 0:
            try:
                result_df["日期"] = pd.to_datetime(df.iloc[:, 0])
            except (ValueError, TypeError):
                print("警告: 未找到日期列")
                return None
        else:
            print("警告: 未找到日期列")
            return None

    # 处理其他列 - 查找对应的列名
    col_mapping = {
        "开盘价": ["开盘", "open", "Open", "OPEN"],
        "收盘价": ["收盘", "close", "Close", "CLOSE", "latest", "Latest", "LATEST"],
        "最高价": ["最高", "high", "High", "HIGH"],
        "最低价": ["最低", "low", "Low", "LOW"],
        "交易量": [
            "成交量",
            "volume",
            "Volume",
            "VOLUME",
            "成交额",
            "amount",
            "Amount",
        ],
    }

    for target_col, possible_cols in col_mapping.items():
        found = False
        for source_col in possible_cols:
            if source_col in df.columns:
                result_df[target_col] = pd.to_numeric(df[source_col], errors="coerce")
                found = True
                break

        if not found:
            # 尝试通过列名包含关键词来查找
            for col in df.columns:
                if target_col.replace("价", "").replace("量", "") in str(col).lower():
                    result_df[target_col] = pd.to_numeric(df[col], errors="coerce")
                    found = True
                    break

        if not found:
            print(f"警告: 未找到 {target_col} 对应的列，将设为空值")
            result_df[target_col] = None

    # 按日期排序
    result_df = result_df.sort_values("日期").reset_index(drop=True)

    # 筛选日期范围
    result_df = result_df[
        (result_df["日期"] >= "2000-01-01") & (result_df["日期"] <= "2019-12-31")
    ]

    # 计算单日涨跌幅：((当前收盘价 - 前一天收盘价) / 前一天收盘价) × 100%
    if "收盘价" in result_df.columns and result_df["收盘价"].notna().any():
        # 使用明确的公式计算
        result_df["单日涨跌幅"] = (
            (result_df["收盘价"] - result_df["收盘价"].shift(1))
            / result_df["收盘价"].shift(1)
            * 100
        )
    else:
        print("警告: 无法计算涨跌幅，因为收盘价数据缺失")
        result_df["单日涨跌幅"] = None

    return result_df


def format_data_for_excel(df):
    """
    格式化数据为Excel输出格式
    字段顺序：日期、收盘、开盘、高、低、交易量、涨跌幅
    """
    if df is None or df.empty:
        return None

    # 创建格式化后的DataFrame
    formatted_df = pd.DataFrame()

    # 1. 格式化日期：2020/1/1 格式（去掉前导零）
    if "日期" in df.columns:
        formatted_df["日期"] = df["日期"].apply(
            lambda x: f"{x.year}/{x.month}/{x.day}" if pd.notna(x) else ""
        )
    else:
        formatted_df["日期"] = ""

    # 2. 格式化价格：收盘、开盘、高（最高价）、低（最低价）
    # 格式为 '1,946.41'（千分位逗号，保留2位小数）
    def format_price(value):
        if pd.isna(value):
            return ""
        return f"{value:,.2f}"

    if "收盘价" in df.columns:
        formatted_df["收盘"] = df["收盘价"].apply(format_price)
    else:
        formatted_df["收盘"] = ""

    if "开盘价" in df.columns:
        formatted_df["开盘"] = df["开盘价"].apply(format_price)
    else:
        formatted_df["开盘"] = ""

    if "最高价" in df.columns:
        formatted_df["高"] = df["最高价"].apply(format_price)
    else:
        formatted_df["高"] = ""

    if "最低价" in df.columns:
        formatted_df["低"] = df["最低价"].apply(format_price)
    else:
        formatted_df["低"] = ""

    # 3. 格式化交易量：B单位，格式为 (num / 1000000000).toFixed(2) + "B"
    # 如果没有交易量数据，用0代替
    if "交易量" in df.columns and df["交易量"].notna().any():
        formatted_df["交易量"] = df["交易量"].apply(
            lambda x: f"{x / 1000000000:.2f}B" if pd.notna(x) and x != 0 else "0"
        )
    else:
        formatted_df["交易量"] = "0"

    # 4. 格式化涨跌幅：百分比格式 "-4.35%"
    if "单日涨跌幅" in df.columns:
        formatted_df["涨跌幅"] = df["单日涨跌幅"].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else ""
        )
    else:
        formatted_df["涨跌幅"] = ""

    return formatted_df


def main():
    """
    主函数
    """
    print("=" * 60)
    print("恒生科技指数数据获取工具 (2000-2019)")
    print("=" * 60)

    # 获取数据
    df = get_hstech_data()

    if df is None or df.empty:
        print("\n无法获取数据，请检查:")
        print("1. 网络连接是否正常")
        print("2. AKShare 版本是否最新")
        print("3. 指数代码是否正确")
        print("\n建议运行以下命令更新 AKShare:")
        print("pip install akshare --upgrade")
        return

    # 处理数据
    print("\n正在处理数据...")
    processed_df = process_data(df)

    if processed_df is None or processed_df.empty:
        print("数据处理失败")
        return

    # 格式化数据为CSV格式
    print("\n正在格式化数据...")
    formatted_df = format_data_for_excel(processed_df)

    if formatted_df is None or formatted_df.empty:
        print("数据格式化失败")
        return

    # 保存为CSV文件
    output_file = "恒生科技指数_2000-2019.csv"
    formatted_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n数据已成功保存至: {output_file}")
    print(f"共 {len(formatted_df)} 条记录")
    print("\n数据预览（前10行）:")
    print(formatted_df.head(10))
    print("\n数据预览（后10行）:")
    print(formatted_df.tail(10))


if __name__ == "__main__":
    main()
