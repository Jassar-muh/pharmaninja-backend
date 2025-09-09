o#!/bin/bash
# test.sh — PharmaNinja backend tests

BASE=https://pharmaninja-backend.onrender.com
SID=test$(date +%s)

echo "==> /ping"
curl -s "$BASE/ping"; echo

echo "==> /health"
curl -s "$BASE/health" | jq .

echo "==> /selftest"
curl -s "$BASE/selftest" | jq .

echo
echo "==> English: Beta-lactam antibiotics"
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"sessionId":"'"$SID"'","lang":"EN","stage":"3rd","subject":"Pharmacology","question":"Explain beta-lactam antibiotics."}' \
  | jq .

echo
echo "==> Follow-up MCQs (same session)"
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"sessionId":"'"$SID"'","question":"MCQs please"}' \
  | jq .

echo
echo "==> Compare with glycopeptides (same session)"
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"sessionId":"'"$SID"'","question":"compare with glycopeptides"}' \
  | jq .

echo
echo "==> Switch to Arabic"
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"sessionId":"'"$SID"'","question":"arabic"}' \
  | jq .

echo
echo "==> Arabic MCQs"
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"sessionId":"'"$SID"'","question":"اعطني 5 اسئلة اختيار من متعدد عنها"}' \
  | jq .
chmod +x test.sh
