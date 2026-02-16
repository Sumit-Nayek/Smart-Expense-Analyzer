import pdfplumber
import re
from datetime import datetime
import pandas as pd
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class TransactionExtractor:
    """
    Extract transaction details from PDFs with the structure shown in the screenshot:
    - Date and Time
    - Recipient (Paid to / Received from)
    - Transaction Type (DEBIT/CREDIT)
    - Amount
    - Transaction ID and UTR (optional)
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.transactions = []

    def extract_transactions(self) -> pd.DataFrame:
        """
        Main method to extract all transactions from PDF
        """
        print(f"\n{'='*60}")
        print(f"Extracting transactions from: {self.pdf_path}")
        print('='*60)

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"\n📄 Processing Page {page_num}...")

                    # Extract text from page
                    text = page.extract_text()
                    if not text:
                        print(f"  ⚠️ No text found on page {page_num}")
                        continue

                    # Split text into lines
                    lines = text.split('\n')

                    # Process the page to find transactions
                    page_transactions = self._process_page(lines, page_num)

                    if page_transactions:
                        self.transactions.extend(page_transactions)
                        print(f"  ✅ Found {len(page_transactions)} transaction(s) on page {page_num}")

            # Create DataFrame
            df = self._create_dataframe()

            # Print summary
            self._print_summary(df)

            return df

        except Exception as e:
            print(f"❌ Error extracting transactions: {str(e)}")
            return pd.DataFrame()

    def _process_page(self, lines: List[str], page_num: int) -> List[Dict]:
        """
        Process a page's lines to extract transaction blocks
        """
        transactions = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Look for transaction start (usually contains date)
            if self._looks_like_date_line(line):
                transaction = self._extract_transaction_block(lines, i, page_num)
                if transaction:
                    transactions.append(transaction)
                    # Print each transaction as it's found
                    self._print_transaction(transaction)

                    # Skip ahead to avoid reprocessing
                    i += 4  # Transaction usually spans 3-4 lines
                else:
                    i += 1
            else:
                i += 1

        return transactions

    def _extract_transaction_block(self, lines: List[str], start_idx: int, page_num: int) -> Optional[Dict]:
        """
        Extract a complete transaction block starting at start_idx
        """
        try:
            transaction = {
                'page_number': page_num,
                'raw_lines': []
            }

            # Line 1: Date and main transaction info
            line1 = lines[start_idx].strip()
            transaction['raw_lines'].append(line1)

            # Extract date from line 1
            date_time = self._extract_date_time(line1)
            if date_time:
                transaction.update(date_time)

            # Extract recipient and type from line 1
            recipient_info = self._extract_recipient_and_type(line1)
            if recipient_info:
                transaction.update(recipient_info)

            # Extract amount from line 1
            amount = self._extract_amount(line1)
            if amount:
                transaction['amount'] = amount

            # Look for time in the next few lines if not found in line1
            if 'time' not in transaction:
                # Check next line for time
                if start_idx + 1 < len(lines):
                    next_line = lines[start_idx + 1].strip()
                    time_match = re.search(r'(\d{1,2}:\d{2}\s*[ap]m)', next_line.lower())
                    if time_match:
                        transaction['time'] = time_match.group(1)
                        transaction['raw_lines'].append(next_line)
                
                # If still not found, check the line after that
                if 'time' not in transaction and start_idx + 2 < len(lines):
                    next_next_line = lines[start_idx + 2].strip()
                    time_match = re.search(r'(\d{1,2}:\d{2}\s*[ap]m)', next_next_line.lower())
                    if time_match:
                        transaction['time'] = time_match.group(1)
                        transaction['raw_lines'].append(next_next_line)

            # Line containing Transaction ID and UTR (if available)
            # Check all subsequent lines for transaction IDs
            for offset in [1, 2, 3]:
                if start_idx + offset < len(lines):
                    check_line = lines[start_idx + offset].strip()
                    if check_line and ('Transaction ID' in check_line or 'UTR' in check_line):
                        if check_line not in transaction['raw_lines']:
                            transaction['raw_lines'].append(check_line)
                        tx_info = self._extract_transaction_ids(check_line)
                        transaction.update(tx_info)
                        break

            # Line with "Paid by" info (if available)
            for offset in [1, 2, 3]:
                if start_idx + offset < len(lines):
                    check_line = lines[start_idx + offset].strip()
                    if check_line and 'Paid by' in check_line:
                        if check_line not in transaction['raw_lines']:
                            transaction['raw_lines'].append(check_line)
                        transaction['paid_by'] = check_line.replace('Paid by', '').strip()
                        break

            return transaction

        except Exception as e:
            print(f"Error extracting transaction block: {e}")
            return None

    def _looks_like_date_line(self, line: str) -> bool:
        """
        Check if line contains a date (like "Feb 13, 2026")
        """
        date_patterns = [
            r'[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}',  # Feb 13, 2026
            r'\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}',   # 13 Feb 2026
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',     # 13/02/2026 or 13-02-2026
        ]

        for pattern in date_patterns:
            if re.search(pattern, line):
                return True
        return False

    def _extract_date_time(self, line: str) -> Dict:
        """
        Extract date from line (time extraction moved to separate logic)
        """
        result = {}

        # Extract date (e.g., "Feb 13, 2026")
        date_match = re.search(r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})', line)
        if date_match:
            result['date'] = date_match.group(1)

            # Try to parse date for sorting
            try:
                parsed_date = datetime.strptime(date_match.group(1), '%b %d, %Y')
                result['parsed_date'] = parsed_date
            except:
                pass

        # Also check for time in the same line (just in case)
        time_match = re.search(r'(\d{1,2}:\d{2}\s*[ap]m)', line.lower())
        if time_match:
            result['time'] = time_match.group(1)

        return result

    def _extract_recipient_and_type(self, line: str) -> Dict:
        """
        Extract recipient and transaction type (DEBIT/CREDIT)
        """
        result = {}

        # Find transaction type
        if 'DEBIT' in line:
            result['type'] = 'DEBIT'
        elif 'CREDIT' in line:
            result['type'] = 'CREDIT'
        elif 'Paid to' in line:
            result['type'] = 'DEBIT'
        elif 'Received from' in line:
            result['type'] = 'CREDIT'

        # Extract recipient
        # Pattern: "Paid to [Recipient Name]"
        recipient_match = re.search(r'Paid to\s+(.+?)(?:\s+DEBIT|\s+CREDIT|$)', line)
        if recipient_match:
            result['recipient'] = recipient_match.group(1).strip()
        else:
            # Try alternative pattern
            recipient_match = re.search(r'Received from\s+(.+?)(?:\s+DEBIT|\s+CREDIT|$)', line)
            if recipient_match:
                result['recipient'] = recipient_match.group(1).strip()

        return result

    def _extract_amount(self, line: str) -> Optional[float]:
        """
        Extract amount from line (handles ₹ symbol)
        """
        # Pattern for amount with ₹ symbol
        amount_match = re.search(r'[₹]\s*(\d+(?:,\d+)*(?:\.\d{2})?)', line)
        if amount_match:
            amount_str = amount_match.group(1).replace(',', '')
            try:
                return float(amount_str)
            except:
                pass

        # Pattern for amount without ₹ symbol but with decimal
        amount_match = re.search(r'(\d+(?:,\d+)*(?:\.\d{2})?)\s*$', line)
        if amount_match:
            amount_str = amount_match.group(1).replace(',', '')
            try:
                return float(amount_str)
            except:
                pass

        return None

    def _extract_transaction_ids(self, line: str) -> Dict:
        """
        Extract Transaction ID and UTR number
        """
        result = {}

        # Extract Transaction ID
        tx_match = re.search(r'Transaction ID\s+(\S+)', line)
        if tx_match:
            result['transaction_id'] = tx_match.group(1)

        # Extract UTR Number
        utr_match = re.search(r'UTR No\.?\s+(\S+)', line)
        if utr_match:
            result['utr_number'] = utr_match.group(1)

        return result

    def _print_transaction(self, transaction: Dict):
        """
        Print individual transaction details
        """
        print(f"\n  {'─'*40}")
        print(f"  📅 Date: {transaction.get('date', 'N/A')} {transaction.get('time', '')}")
        print(f"  👤 Recipient: {transaction.get('recipient', 'N/A')}")
        print(f"  💳 Type: {transaction.get('type', 'N/A')}")
        print(f"  💰 Amount: ₹{transaction.get('amount', 0):,.2f}")

        if transaction.get('transaction_id'):
            print(f"  🆔 Transaction ID: {transaction['transaction_id']}")
        if transaction.get('utr_number'):
            print(f"  🔢 UTR: {transaction['utr_number']}")
        

    def _create_dataframe(self) -> pd.DataFrame:
        """
        Create pandas DataFrame from extracted transactions
        """
        if not self.transactions:
            return pd.DataFrame()

        df = pd.DataFrame(self.transactions)

        # Select and order relevant columns
        columns = ['date', 'time', 'recipient', 'type', 'amount',
                  'transaction_id', 'utr_number']

        # Keep only columns that exist
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]

        # Sort by date if available
        if 'parsed_date' in df.columns:
            df = df.sort_values('parsed_date')
        elif 'date' in df.columns:
            df = df.sort_values('date')

        return df

    def _print_summary(self, df: pd.DataFrame):
        """
        Print summary of all extracted transactions
        """
        if df.empty:
            print("\n❌ No transactions found in the PDF")
            return

        print(f"\n{'='*60}")
        print(f"📊 EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total transactions found: {len(df)}")

        # Calculate totals
        if 'amount' in df.columns and 'type' in df.columns:
            total_debit = df[df['type'] == 'DEBIT']['amount'].sum()
            total_credit = df[df['type'] == 'CREDIT']['amount'].sum()

            print(f"\n💰 Financial Summary:")
            print(f"  Total Debit:  ₹{total_debit:,.2f}")
            print(f"  Total Credit: ₹{total_credit:,.2f}")
            print(f"  Net Flow:     ₹{total_credit - total_debit:,.2f}")

        # Date range
        if 'date' in df.columns:
            dates = df['date'].dropna()
            if not dates.empty:
                print(f"\n📅 Date Range: {dates.min()} to {dates.max()}")

        print(f"\n📋 First 3 transactions:")
        print(df.head(3).to_string(index=False))

        # Save to CSV option
        print(f"\n💾 To save to CSV: df.to_csv('transactions.csv', index=False)")

# Function to extract from multiple PDFs
def extract_from_multiple_pdfs(pdf_paths: List[str]) -> pd.DataFrame:
    """
    Extract transactions from multiple PDF files
    """
    all_transactions = []

    for pdf_path in pdf_paths:
        extractor = TransactionExtractor(pdf_path)
        df = extractor.extract_transactions()
        if not df.empty:
            df['source_file'] = pdf_path
            all_transactions.append(df)

    if all_transactions:
        combined_df = pd.concat(all_transactions, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

# Main execution
if __name__ == "__main__":
    # Example usage
    pdf_file = "/content/PhonePe_Statement_Aug2025_Sept2025.pdf"  # Replace with your PDF file

    try:
        # Extract transactions
        extractor = TransactionExtractor(pdf_file)
        df = extractor.extract_transactions()

        if not df.empty:
            # Additional analysis
            print(f"\n{'='*60}")
            print("🔍 ADDITIONAL ANALYSIS")
            print('='*60)

            # Group by recipient
            if 'recipient' in df.columns and 'amount' in df.columns:
                print("\n💰 Top Recipients (by total amount):")
                recipient_totals = df.groupby('recipient')['amount'].sum().sort_values(ascending=False).head(5)
                for recipient, amount in recipient_totals.items():
                    print(f"  {recipient}: ₹{amount:,.2f}")

            # Transactions by type
            if 'type' in df.columns:
                print("\n📊 Transactions by Type:")
                type_counts = df['type'].value_counts()
                for t_type, count in type_counts.items():
                    print(f"  {t_type}: {count} transactions")

            # Save to CSV
            csv_file = pdf_file.replace('.pdf', '_transactions.csv')
            df.to_csv(csv_file, index=False)
            print(f"\n💾 Saved to: {csv_file}")

            # Save to Excel with formatting
            excel_file = pdf_file.replace('.pdf', '_transactions.xlsx')
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Transactions', index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets['Transactions']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            print(f"💾 Saved to: {excel_file}")

    except FileNotFoundError:
        print(f"❌ File '{pdf_file}' not found. Please provide a valid PDF path.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")