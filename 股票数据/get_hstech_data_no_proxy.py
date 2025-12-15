"""
获取恒生科技指数数据 - 禁用代理版本
"""

import os

# 清除所有代理设置
for key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
]:
    if key in os.environ:
        del os.environ[key]

import akshare as ak
import pandas as pd
import time

# 日期范围配置
# 注意：恒生科技指数接口实际最早数据从2014年12月31日开始
# 接口不包含2015年之前的历史数据
START_DATE = "2000-01-01"  # 期望的开始日期（但接口可能不包含此日期之前的数据）
END_DATE = "2019-12-31"

# 设置requests不使用代理
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 创建session并禁用代理
session = requests.Session()
session.proxies = {}
adapter = HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1))
session.mount("http://", adapter)
session.mount("https://", adapter)

# 尝试设置akshare使用我们的session（如果支持）
try:
    import akshare.tool.tool_api as tool_api
    # 这里可能需要根据akshare的实际实现来设置
except:
    pass


def get_hstech_data():
    """
    获取恒生科技指数历史数据（2000-2019）
    注意：接口可能不包含2015年之前的数据，恒生科技指数实际数据最早从2014年12月31日开始
    """
    print("正在获取恒生科技指数数据（2000-2019）...")
    print("已禁用代理设置")

    try:
        # 方法2: 尝试使用 stock_hk_index_daily_em 接口
        print("尝试方法: stock_hk_index_daily_em...")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"  尝试 {attempt + 1}/{max_retries}...")
                df = ak.stock_hk_index_daily_em(symbol="HSTECH")

                # 识别日期列
                date_col = None
                if "date" in df.columns:
                    date_col = "date"
                elif "日期" in df.columns:
                    date_col = "日期"
                else:
                    date_col = df.columns[0]

                # 转换日期格式
                df[date_col] = pd.to_datetime(df[date_col])

                # 检查接口返回的实际数据范围
                actual_start = df[date_col].min()
                actual_end = df[date_col].max()
                print(
                    f"接口返回的数据范围: {actual_start.strftime('%Y-%m-%d')} 至 {actual_end.strftime('%Y-%m-%d')}"
                )
                print(f"接口返回的总数据条数: {len(df)}")

                # 如果接口返回的最早日期晚于请求的开始日期，给出提示
                if actual_start > pd.to_datetime(START_DATE):
                    print(
                        f"\n⚠️  警告: 接口返回的最早数据日期为 {actual_start.strftime('%Y-%m-%d')}"
                    )
                    print(
                        f"   请求的开始日期为 {START_DATE}，接口没有 {actual_start.strftime('%Y-%m-%d')} 之前的数据"
                    )
                    print(
                        f"   恒生科技指数可能是在 {actual_start.strftime('%Y年%m月')} 才开始有数据的\n"
                    )

                # 筛选日期范围
                df = df[(df[date_col] >= START_DATE) & (df[date_col] <= END_DATE)]

                print(f"筛选后数据条数: {len(df)} 条记录")
                if len(df) > 0:
                    print(
                        f"筛选后的日期范围: {df[date_col].min().strftime('%Y-%m-%d')} 至 {df[date_col].max().strftime('%Y-%m-%d')}"
                    )

                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"  尝试 {attempt + 1} 失败: {type(e).__name__}")
                    print(f"  等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"方法失败: {e}")
                    return None
    except Exception as e:
        print(f"获取数据时出错: {e}")
        return None

    return None


def process_data(df):
    """
    处理原始数据，转换为标准格式
    """
    if df is None or df.empty:
        return None

    result_df = pd.DataFrame()

    # 识别日期列
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
        except Exception as e:
            print(f"无法识别日期列: {e}")
            return None

    # 处理价格列
    col_mapping = {
        "收盘价": ["收盘", "close", "Close", "CLOSE", "latest", "Latest", "LATEST"],
        "开盘价": ["开盘", "open", "Open", "OPEN"],
        "最高价": ["最高", "high", "High", "HIGH"],
        "最低价": ["最低", "low", "Low", "LOW"],
        "交易量": ["交易量", "volume", "Volume", "VOLUME", "成交量"],
    }

    for target_col, possible_cols in col_mapping.items():
        found = False
        for source_col in possible_cols:
            if source_col in df.columns:
                result_df[target_col] = pd.to_numeric(df[source_col], errors="coerce")
                found = True
                break

        if not found:
            # 如果找不到交易量，设置为0或空
            if target_col == "交易量":
                result_df[target_col] = 0
            else:
                print(f"警告: 未找到 {target_col} 对应的列")

    # 按日期排序
    result_df = result_df.sort_values("日期").reset_index(drop=True)

    # 计算涨跌幅
    if "收盘价" in result_df.columns and result_df["收盘价"].notna().any():
        result_df["涨跌幅"] = result_df["收盘价"].pct_change() * 100
    else:
        result_df["涨跌幅"] = None

    return result_df


def format_data_for_excel(df):
    """
    格式化数据为Excel/CSV输出格式
    """
    if df is None or df.empty:
        return None

    formatted_df = pd.DataFrame()

    # 日期格式：YYYY/M/D
    formatted_df["日期"] = df["日期"].apply(
        lambda x: f"{x.year}/{x.month}/{x.day}" if pd.notna(x) else ""
    )

    # 价格格式：带千分位，保留2位小数
    def format_price(value):
        if pd.isna(value):
            return ""
        return f"{value:,.2f}"

    formatted_df["收盘"] = (
        df["收盘价"].apply(format_price) if "收盘价" in df.columns else ""
    )
    formatted_df["开盘"] = (
        df["开盘价"].apply(format_price) if "开盘价" in df.columns else ""
    )
    formatted_df["高"] = (
        df["最高价"].apply(format_price) if "最高价" in df.columns else ""
    )
    formatted_df["低"] = (
        df["最低价"].apply(format_price) if "最低价" in df.columns else ""
    )

    # 交易量
    if "交易量" in df.columns:
        formatted_df["交易量"] = df["交易量"].apply(
            lambda x: int(x) if pd.notna(x) else 0
        )
    else:
        formatted_df["交易量"] = 0

    # 涨跌幅格式：保留2位小数，带百分号
    if "涨跌幅" in df.columns:
        formatted_df["涨跌幅"] = df["涨跌幅"].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else ""
        )
    else:
        formatted_df["涨跌幅"] = ""

    return formatted_df


def main():
    print("=" * 60)
    print("恒生科技指数数据获取工具 (2000-2019) - 无代理版本")
    print("=" * 60)

    df = get_hstech_data()

    if df is None or df.empty:
        print("\n无法获取数据")
        return

    # 处理数据
    print("\n正在处理数据...")
    processed_df = process_data(df)

    if processed_df is None or processed_df.empty:
        print("数据处理失败")
        return

    # 格式化数据
    print("\n正在格式化数据...")
    formatted_df = format_data_for_excel(processed_df)

    if formatted_df is None or formatted_df.empty:
        print("数据格式化失败")
        return

    # 保存为CSV文件
    output_file = f"恒生科技指数_{START_DATE[:4]}-{END_DATE[:4]}.csv"
    formatted_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n数据已成功保存至: {output_file}")
    print(f"共 {len(formatted_df)} 条记录")
    print("\n数据预览（前10行）:")
    print(formatted_df.head(10))


if __name__ == "__main__":
    main()
