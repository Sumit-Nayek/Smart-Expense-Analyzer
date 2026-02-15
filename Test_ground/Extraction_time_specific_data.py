import pdfplumber
import re
from datetime import datetime
import pandas as pd
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class TransactionExtractor:
    """
    Extract transaction details from PDF statements.
    Handles date, time, recipient, type (DEBIT/CREDIT), amount, and reference numbers.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.transactions = []

    def extract_transactions(self) -> pd.DataFrame:
        """Main extraction routine – processes all pages and returns a DataFrame."""
        print(f"\n{'='*60}\nExtracting transactions from: {self.pdf_path}\n{'='*60}")

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"\n📄 Page {page_num}...")
                    text = page.extract_text()
                    if not text:
                        continue

                    lines = text.split('\n')
                    page_txs = self._process_page(lines, page_num)
                    self.transactions.extend(page_txs)
                    print(f"  ✅ Found {len(page_txs)} transaction(s)")

            df = self._create_dataframe()
            self._print_summary(df)
            return df

        except Exception as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()

    def _process_page(self, lines: List[str], page_num: int) -> List[Dict]:
        """Scan lines and group them into transaction blocks."""
        transactions = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if self._is_date_line(line):
                tx = self._build_transaction(lines, i, page_num)
                if tx:
                    transactions.append(tx)
                    self._print_transaction(tx)
                    i += 3  # typical block length
                else:
                    i += 1
            else:
                i += 1
        return transactions

    def _is_date_line(self, line: str) -> bool:
        """Detect if a line likely starts a transaction (contains a date)."""
        return bool(re.search(r'(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})', line))

    def _build_transaction(self, lines: List[str], start: int, page_num: int) -> Optional[Dict]:
        """Assemble a transaction dictionary from consecutive lines."""
        try:
            tx = {'page': page_num}
            # Line 1: main info
            line1 = lines[start].strip()
            tx.update(self._extract_date_time(line1))
            tx.update(self._extract_recipient_type(line1))
            tx['amount'] = self._extract_amount(line1)

            # Line 2: transaction IDs
            if start + 1 < len(lines):
                line2 = lines[start + 1].strip()
                if 'Transaction ID' in line2 or 'UTR' in line2:
                    tx.update(self._extract_ids(line2))

            # Line 3: paid by info
            if start + 2 < len(lines):
                line3 = lines[start + 2].strip()
                if 'Paid by' in line3:
                    tx['paid_by'] = line3.replace('Paid by', '').strip()

            return tx
        except Exception:
            return None

    def _extract_date_time(self, line: str) -> Dict:
        """Extract date and time from a line (they may be on separate lines)."""
        result = {}

        # Date patterns
        date_match = re.search(r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})', line) or \
                     re.search(r'(\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4})', line)
        if date_match:
            result['date'] = date_match.group(1)
            try:
                # Parse for sorting
                result['parsed_date'] = datetime.strptime(date_match.group(1), '%b %d, %Y')
            except:
                pass

        # Time pattern – case‑insensitive, optional space
        time_match = re.search(r'(\d{1,2}:\d{2}\s*[AP]M?)', line, re.IGNORECASE)
        if time_match:
            result['time'] = time_match.group(1)
        else:
            # Sometimes time is on the next line (e.g., after date)
            # We'll handle it later if needed – you could extend this method to accept a second line.
            pass

        return result

    def _extract_recipient_type(self, line: str) -> Dict:
        """Extract recipient name and transaction type."""
        result = {}
        # Type keywords
        if 'DEBIT' in line:
            result['type'] = 'DEBIT'
        elif 'CREDIT' in line:
            result['type'] = 'CREDIT'
        elif 'Paid to' in line:
            result['type'] = 'DEBIT'
        elif 'Received from' in line:
            result['type'] = 'CREDIT'

        # Recipient patterns
        rec_match = re.search(r'(?:Paid to|Received from|To|From|Beneficiary[:\s])\s+(.+?)(?:\s+(?:DEBIT|CREDIT|DR|CR)|$)', line, re.IGNORECASE)
        if rec_match:
            result['recipient'] = rec_match.group(1).strip()

        return result

    def _extract_amount(self, line: str) -> Optional[float]:
        """Extract numeric amount (handles ₹, commas, decimals)."""
        match = re.search(r'[₹]?\s*(\d+(?:,\d+)*(?:\.\d{2})?)', line)
        if match:
            return float(match.group(1).replace(',', ''))
        return None

    def _extract_ids(self, line: str) -> Dict:
        """Extract Transaction ID and UTR number."""
        result = {}
        tx = re.search(r'Transaction ID\s+(\S+)', line)
        if tx:
            result['transaction_id'] = tx.group(1)
        utr = re.search(r'UTR No\.?\s+(\S+)', line)
        if utr:
            result['utr_number'] = utr.group(1)
        return result

    def _print_transaction(self, tx: Dict):
        """Pretty‑print a single transaction."""
        print(f"\n  {'─'*40}")
        print(f"  📅 {tx.get('date', 'N/A')} {tx.get('time', '')}")
        print(f"  👤 Recipient: {tx.get('recipient', 'N/A')}")
        print(f"  💳 Type: {tx.get('type', 'N/A')}")
        print(f"  💰 Amount: ₹{tx.get('amount', 0):,.2f}")
        if 'transaction_id' in tx:
            print(f"  🆔 ID: {tx['transaction_id']}")
        if 'utr_number' in tx:
            print(f"  🔢 UTR: {tx['utr_number']}")
        if 'paid_by' in tx:
            print(f"  💳 Paid by: {tx['paid_by']}")

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert transactions to DataFrame and sort."""
        if not self.transactions:
            return pd.DataFrame()
        df = pd.DataFrame(self.transactions)
        # Keep relevant columns
        cols = ['date', 'time', 'recipient', 'type', 'amount',
                'transaction_id', 'utr_number', 'paid_by', 'page']
        df = df[[c for c in cols if c in df.columns]]
        if 'parsed_date' in df.columns:
            df = df.sort_values('parsed_date').drop(columns='parsed_date')
        return df

    def _print_summary(self, df: pd.DataFrame):
        """Display summary statistics."""
        if df.empty:
            print("\n❌ No transactions found.")
            return
        print(f"\n{'='*60}\n📊 SUMMARY\n{'='*60}")
        print(f"Total transactions: {len(df)}")
        if 'amount' in df and 'type' in df:
            debit = df[df['type'] == 'DEBIT']['amount'].sum()
            credit = df[df['type'] == 'CREDIT']['amount'].sum()
            print(f"Total Debit:  ₹{debit:,.2f}\nTotal Credit: ₹{credit:,.2f}\nNet: ₹{credit - debit:,.2f}")
        if 'date' in df:
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))
        print("\n💾 Save with: df.to_csv('transactions.csv', index=False)")

# Convenience function
def extract_from_pdf(pdf_path: str) -> pd.DataFrame:
    return TransactionExtractor(pdf_path).extract_transactions()

if __name__ == "__main__":
    pdf_file = "/content/PhonePe_Statement_Aug2025_Sept2025.pdf"
    df = extract_from_pdf(pdf_file)
    if not df.empty:
        df.to_csv(pdf_file.replace('.pdf', '_transactions.csv'), index=False)
        print(f"\n✅ Saved CSV.")