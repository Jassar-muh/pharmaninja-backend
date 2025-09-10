#!/bin/bash
# PharmaNinja backend test script

BASE=${BASE:-https://pharmaninja-backend.onrender.com}
SID=test1

echo "==> /ping"
curl -s "$BASE/ping"
echo -e "\n"

echo "==> /health"
curl -s "$BASE/health"
echo -e "\n"

echo "==> /selftest"
curl -s "$BASE/selftest"
echo -e "\n"

echo "==> English: Beta-lactam antibiotics"
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"sessionId":"'"$SID"'","lang":"EN","stage":"3rd","subject":"Pharmacology","question":"Explain beta-lactam antibiotics."}'
echo -e "\n"

echo "==> Follow-up MCQs (same session)"
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"sessionId":"'"$SID"'","question":"MCQs please"}'
echo -e "\n"

echo "==> Compare with glycopeptides (same session)"
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"sessionId":"'"$SID"'","question":"Compare with glycopeptides"}'
echo -e "\n"

echo "==> Switch to Arabic"
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"sessionId":"'"$SID"'","question":"arabic"}'
echo -e "\n"

echo "==> Arabic MCQs"
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"sessionId":"'"$SID"'","question":"اعطني 5 اسئلة اختيار من متعدد"}'
echo -e "\n"
