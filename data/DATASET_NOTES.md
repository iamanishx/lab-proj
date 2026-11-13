# Scam Encounters India - Dataset Notes

Generated 13 Nov 2026 (IST). Companion to scam_encounters_india.csv.

## Size
- 3,005 rows total: 3,000 scraped scam encounters plus 5 survey responses.
- The 5 survey rows contain the respondents’ reported daily UPI-use answers; the 3,000 scraped rows have this value as unknown.
- The 3,000 scraped rows have 1,638 distinct published source URLs. The 5 survey rows use `source_url=survey`. Scraped rows sharing a URL are distinct incidents reported in the same complaint thread (original post + commenters), each with its own details.

## Schema (exact column order)
channel (call/sms/whatsapp/email), claimed_sender (bank_kyc/refund/prize/job/family_emergency/other), asked_for (otp/pin/link/qr_scan/money/nothing), urgency_pressure (yes/no), person_daily_upi_use (unknown for every scraped row), engaged (ignored/replied/clicked/shared_otp/lost_money), source_url.

## Class balance (3,000 scraped rows only; excludes the 5 survey responses)
- channel: call 1266 (42.2%), email 578 (19.3%), whatsapp 1000 (33.3%), sms 156 (5.2%)
- claimed_sender: other 1217, prize 774 (25.8%), job 645, bank_kyc 224, refund 88, family_emergency 52
- asked_for: money 2037, nothing 589, link 117, otp 111, qr_scan 77, pin 69
- urgency_pressure: no 2558, yes 442
- engaged: ignored 734, replied 945, lost_money 1246, clicked 28, shared_otp 47

## Sources (the 3,000 scraped rows come from individually published public reports, each traceable via source_url; the 5 survey rows are marked `source_url=survey`)
- ConsumerComplaints.in individual complaint pages and their comment threads (2229 rows): first-person scam reports across KBC/lottery, fake bank/KYC calls, OLX/army QR scams, data-entry job scams, gigolo/escort job scams, donation calls, challan/legal-threat SMS, fake bank SMS, loan-app extortion, digital arrest, courier/customs, insurance bonus/refund calls.
- Old-style ConsumerComplaints megathreads (KBC lottery, TCS/visa) rule-coded per comment: these carry ~10-15% residual noise and skew prize/lottery.
- Reddit (r/india, r/legaladviceindia, r/scams and similar) first-person posts, hand-coded: high confidence.

## Coding conventions
- Telegram/Instagram/Snapchat-origin messaging scams are coded whatsapp (schema has no telegram); flagged here as a known approximation.
- engaged = most severe outcome wins; asked_for priority otp > qr_scan > pin > link > money > nothing.
- bank_kyc covers bank/wallet/KYC/card impersonation; card/identity-detail asks coded as pin; sharing identity documents (Aadhaar) coded engaged=shared_otp.
- urgency_pressure=yes only on an explicit threat or deadline (legal action, police, expiry, account freeze).
- Marketplace-originated scams (OLX/99acres/magicbricks) coded by the actual contact channel.
- Multiple incidents in one thread URL are separate rows; bare "same thing happened to me" comments without detail are excluded; one incident reposted in two threads counts once.
- Excluded: recovery-scam ads, fake "customer care" replies by scammers, generic warnings/PSAs with no first-person encounter, pure commercial disputes (undelivered products) without a scam approach, non-Indian reporters, in-person-only frauds with no call/sms/whatsapp/email channel.
- person_daily_upi_use is "unknown" for all 3,000 scraped rows because public reports do not state the victim's daily UPI use. The 5 survey rows contain reported values for this field.

## Known caveats
- Prize is still the largest claimed_sender class (~26%) because of the KBC/lottery megathreads; the share keeps falling as new hand-coded sources are added (was ~41% at 1,750 rows).
- Rule-coded megathread rows carry ~10-15% residual noise; hand-coded rows are high confidence.
- Self-reported public complaints skew toward engaged victims (replied/lost_money) relative to the general population.
