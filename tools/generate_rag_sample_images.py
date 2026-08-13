from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "rag_sources" / "images"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def draw_document(filename: str, title: str, body: str) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (960, 640), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    body_font = load_font(20)
    small_font = load_font(16)

    draw.rectangle((0, 0, 960, 74), fill=(239, 246, 255))
    draw.text((36, 20), title, font=title_font, fill=(15, 23, 42))

    y = 110
    for line in body.splitlines():
        draw.text((48, y), line, font=body_font, fill=(17, 24, 39))
        y += 44

    draw.line((48, y + 18, 912, y + 18), fill=(226, 232, 240), width=2)
    draw.text(
        (48, y + 48),
        "用途：RAG OCR 测试原始图片，非正式业务文件。",
        font=small_font,
        fill=(100, 116, 139),
    )
    image.save(IMAGE_DIR / filename)


def main() -> None:
    docs = [
        (
            "collection_letter_shop093.png",
            "催款函",
            "致：跃动工场 SHOP_093\n"
            "欠费金额：66000 元\n"
            "欠费天数：72 天\n"
            "保证金覆盖率：48%\n"
            "请于 2026 年 8 月 5 日前完成付款。\n"
            "逾期将进入法务复核和合同风险专项评审。",
        ),
        (
            "inspection_photo_note_shop056.png",
            "巡店照片记录",
            "门店：谷雨面馆 SHOP_056\n"
            "现场情况：高峰期排队长，后厨人员不足。\n"
            "投诉焦点：出餐慢、菜品稳定性不足。\n"
            "建议：3 个工作日内提交整改计划。",
        ),
        (
            "closure_risk_photo_shop093.png",
            "疑似撤店现场照片说明",
            "门店：跃动工场 SHOP_093\n"
            "观察：货架空置，体验区设备关闭。\n"
            "水电费波动：-45%\n"
            "建议：楼层经理当天复核营业状态。",
        ),
        (
            "contract_clause_screenshot.png",
            "合同条款截图",
            "条款：保证金扣划与合同解除\n"
            "保证金覆盖率低于 60% 时，应触发财务与法务复核。\n"
            "涉及锁铺、清场、品牌替换的动作必须人工审批。",
        ),
        (
            "arrears_table_screenshot.png",
            "欠费台账截图",
            "SHOP_093 跃动工场 欠费 66000 元 72 天\n"
            "SHOP_083 净衣坊洗护 欠费 52000 元 42 天\n"
            "SHOP_056 谷雨面馆 欠费 35082 元 58 天",
        ),
    ]
    for filename, title, body in docs:
        draw_document(filename, title, body)
    print(f"created {len(docs)} images in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
