"""Browser test with proper CSS waiting."""
import time
from playwright.sync_api import sync_playwright

CSV_PATH = r"C:\Users\bedir\Desktop\ornek_yorumlar.csv"
SS_DIR = r"C:\Users\bedir\Desktop"

def ss(page, name):
    path = f"{SS_DIR}\\test_{name}.png"
    page.screenshot(path=path)
    print(f"  [ss] {name}")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": 1400, "height": 900})
    page = ctx.new_page()

    # 1. Landing
    print("=== Landing ===")
    page.goto("http://localhost:3000", wait_until="networkidle")
    page.wait_for_timeout(3000)  # wait for CSS hydration

    display = page.evaluate(
        "window.getComputedStyle(document.querySelector('input[type=file]')).display"
    )
    print(f"  file input display = '{display}' (want 'none')")

    nav_flex = page.evaluate(
        "window.getComputedStyle(document.querySelector('nav div')).display"
    )
    print(f"  nav div display = '{nav_flex}' (want 'flex')")

    h1 = page.locator("h1").inner_text()
    print(f"  H1: {h1[:60]}")
    ss(page, "01_landing")

    # 2. Upload
    print("\n=== Uploading CSV ===")
    page.locator("input[type='file']").set_input_files(CSV_PATH)
    page.wait_for_timeout(1000)
    ss(page, "02_loading")

    page.wait_for_selector("text=Yapay Zeka Özeti", timeout=120000)
    page.wait_for_timeout(2000)
    ss(page, "03_results_top")

    # Scroll through
    for scroll, name in [(700, "04_score"), (1400, "05_categories"), (2200, "06_procon"), (3200, "07_featured"), (4500, "08_table")]:
        page.evaluate(f"window.scrollTo(0, {scroll})")
        page.wait_for_timeout(600)
        ss(page, name)

    # Search
    print("\n=== Search ===")
    page.locator("input[placeholder*='ara']").fill("kargo")
    page.wait_for_timeout(700)
    ss(page, "09_search")
    rows = page.evaluate("document.querySelectorAll('tbody tr').length")
    print(f"  kargo results: {rows}")
    page.locator("input[placeholder*='ara']").fill("")

    # Filter
    page.locator("button", has_text="Olumsuz").last.click()
    page.wait_for_timeout(700)
    ss(page, "10_olumsuz")
    page.locator("button", has_text="Tümü").last.click()

    # Yeni Analiz
    print("\n=== Yeni Analiz ===")
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(400)
    page.locator("button", has_text="Yeni Analiz").click()
    page.wait_for_timeout(1000)
    ss(page, "11_landing_again")

    print("\n=== DONE ===")
    browser.close()
