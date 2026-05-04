import requests, re

r = requests.get("http://localhost:3001", timeout=10)
print("Status:", r.status_code)
css = re.findall(r'href="([^"]*\.css[^"]*)"', r.text)
print("CSS files:", css[:5])
# Check if Tailwind classes are in the page
print("Has 'flex' class in HTML:", "flex" in r.text)
print("Has 'grid' class in HTML:", "grid" in r.text)

# Fetch one CSS file and check for flex
if css:
    css_url = "http://localhost:3001" + css[0]
    cr = requests.get(css_url, timeout=10)
    print("\nCSS file size:", len(cr.text), "bytes")
    print("Has '.flex' in CSS:", ".flex{" in cr.text or ".flex {" in cr.text)
    print("First 500 chars:", cr.text[:500])
