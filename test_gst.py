"""
test_gst.py

Standalone tester for the GST extraction pipeline.

Usage:
    python test_gst.py

Then enter a company name.

Shows:
- URLs retrieved
- Downloaded pages
- Extracted GST
"""





from identifier_lookup import get_company_identifiers

def main():
    company = input("Company name: ").strip()

    if not company:
        print("No company entered.")
        return

    print("\n" + "=" * 80)
    print(f"Searching for: {company}")
    print("=" * 80)


    result, trace = get_company_identifiers(company)

    print("=" * 80)


    gst = result.get("GST", "GST not found")
    cin = result.get("CIN", "CIN not found")

    print("GST")
    print(gst)

    print("\nCIN")
    print(cin)

    print("=" * 80)
    print("Trace")
    print(f"Corroboration counts: {trace.corroboration_counts}")
    print(f"Validation notes: {trace.validation_notes}")
    print(f"Sites checked: {len(trace.site_checks)}")
    print("=" * 80)

if __name__ == "__main__":
    main()