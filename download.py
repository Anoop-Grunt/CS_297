from pathlib import Path
import requests
import zipfile

URL = "https://cic-dataset.s3.us-east-1.amazonaws.com/cic_dgg_2025.zip?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjELD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQC9%2BY1pcYqUc1iMb6JSD9llEucRpcoOhDAGAefDuG%2BAeQIgO10qTOaNPHbBvIrcOd4smZ3JCj6dO6S1uSy2HXtXEnwquQMIeBAAGgw3ODM3NjQ1Nzk4MTAiDFqt7cJMlTaD6p6BAyqWA7lHw%2Fcn1H8LVxjcFnfr2kG7Ivi%2Bp%2FjYerubXBSv0xsEjYMFkf2Z0FWPrMRXC6gZT4%2BzoSxzpQx2wqdqfWtaq%2Fs5uXSchBUJgmLGz6W57DoJ2WpLJoTqWU5LkOTkf6MYECBFPopk0Vo1Q02nynTFb3pOGmxqkOvIuF34Kmk779sKdQP1WuzRkhbsb2EZX4rXAd8XDRVqmiWbvechQaj7GJ4uf44ikpMmsFm2WPGw8iDNTg2Px%2FR7PX80npwZY%2F4mi5tkT3hUVzdubpRkInGe9LjlB20sS53RfDgoCI3Y28%2Bincde7BkURq%2B%2BmnW1bNvPCF%2BhLVJcGKAGDaZRToMF0iroFHm4xbymXvBxgCCyo2D%2B9Yybq8oThoNV4Ppjv%2BT6NWR5aPgvkwepa%2FFWCrEP6cS4RLbp2f3B4urcoWGE3dtXzWIoSzcX0MheviEDVhtIE%2FtV6T4l0squoTFV1VWJE1dmaAoQogxfIjb%2FcHYjvKaw3QUPHkEG%2Fq%2Bv0zqKlXu%2BzP1yjJpT9mWoms%2BumF0mHMtpkMOldKgw0LKw1gY63gJO5qWq54IkueCOb57J6BD45xe%2BTVKgJQtj9KPaoTeAoG7OcWUkUNHYLEoKbLsUTZ8%2BDc3rH8KbBbHuc1DzuN1kWXdohKjJqlH3CewiSMSftm5so6tnRDSIcPr%2FnSdxXJc4KbAv7CSMi%2FOzLVMe4inVWVYidmZEeg8vYJK15Tes4m1w2phcHfGK5ujgJEtXd4HnQvbUkOLz0N4uF5i3tuFOtkN5bcf2uL6zzD0dUnON4Gx0ZAe3I30%2FT%2FBu6jndZ9O13C%2FeBHgxGsCxf1KpbL8kq9Os9LM1MoH6fQHP6Z3FEJiVbf3wJbJtBP%2F4edrCHDSrpfBIoNEl64V%2B%2BrX4A3yzlxJJ4jkMJubKx1kbx19w4ktjNYYs5Tx45G4ecNCYcmREraOCwIedR8VyxpZ293CmpoJJxoI751n1Y3vUC9YIFc%2BdkKZqcVKU5Mo9aAtedwrSRa08CIbI8VPLBfmUpw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA3M7ACMXRLA65AMYV%2F20261011%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20261011T231907Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=d5c0ffad975685f01492aa1b0497cef2ce8887741abe57f9ffa3a2439b7fb492"
OUTPUT = Path("cic_dgg_2025.zip")

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
}

response = requests.get(
    URL,
    headers=headers,
    stream=True,
    allow_redirects=True,
    timeout=120,
)

print("Status:", response.status_code)
print("Final URL:", response.url)
print("Content-Type:", response.headers.get("content-type"))

response.raise_for_status()

with open(OUTPUT, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)

print(f"Downloaded {OUTPUT.stat().st_size / 1e9:.2f} GB")

if not zipfile.is_zipfile(OUTPUT):
    print("First bytes:", OUTPUT.read_bytes()[:100])
    raise RuntimeError("The response was not a valid ZIP file.")

print("Valid ZIP downloaded.")
