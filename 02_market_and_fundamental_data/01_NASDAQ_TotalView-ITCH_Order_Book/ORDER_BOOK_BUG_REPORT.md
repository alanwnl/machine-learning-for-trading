# NASDAQ ITCH Order Book Reconstruction — Bug Report

**Date**: 2026-03-17
**Affected files**: `02_rebuild_nasdaq_order_book.py`, `02_rebuild_nasdaq_order_book_2.py`, `02_rebuild_nasdaq_order_book_3.py`
**Stock tested**: AMZN (Oct 30, 2019, market price ~$1762)
**Origin**: Bugs exist in the original [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) repository

---

## Symptom

The `plot_order_book_profile` output showed **buy prices higher than sell prices**, violating the fundamental order book invariant that `best_bid < best_ask`:

```
==target buy==   (should be BELOW market price)
price: 1779.02, 1779.01, 1778.83, ...   ← $17 ABOVE market

==target sell==  (should be ABOVE market price)
price: 1760.23, 1761.34, 1761.37, ...   ← $2 BELOW market
```

Buyers were appearing to pay more than sellers — an impossible condition in a real order book.

---

## Root Cause Analysis

Three bugs were identified in the `get_messages()` function and the order book reconstruction loop. All three compound to corrupt the `current_orders` data structure over the course of a trading day.

### Bug 1: Chained U (Replace) Messages Silently Fail

**Location**: `get_messages()` — order lookup table construction

**What happened**: The order lookup table was built only from `A` (Add Order) and `F` (Add Order with MPID) messages:

```python
# OLD (buggy)
orders = pd.concat([data['A'], data['F']], ...).loc[:, order_cols]
```

When a `U` (Replace) message replaces order A→B, the new order reference number B is **never added** to the lookup table. If a second `U` message later replaces B→C, the merge on `original_order_reference_number = B` produces NaN for all fields.

In the main loop, `np.isnan(message.buy_sell_indicator)` catches this and **silently skips** the message. The old order B's shares are never removed, and C's shares are never added.

**Impact**: Over a trading day with thousands of chained replacements, phantom orders accumulate at stale price levels on both sides of the book.

**ITCH specification context**: The `U` message spec states:

> *"The new reference number for this order at time of replacement. Please note that the Nasdaq system will use this new order reference number for all subsequent updates."*

This means any subsequent E, C, X, D, or U message referencing the new order ref will fail to find it.

---

### Bug 2: E (Execution) Messages Remove Total Shares Instead of Executed Shares

**Location**: Main reconstruction loop — `E` message handling

**What happened**:

```python
# OLD (buggy)
if message.type in ['E', 'C', 'X', 'D', 'U']:
    ...
    else:
        shares = -int(message.shares)  # ← original order's TOTAL shares
```

The `message.shares` field came from the order lookup merge, which returns the **original order's total share count**. The ITCH `E` message carries an `executed_shares` field that indicates how many shares were actually filled — which can be a partial fill.

**Example**: Order for 500 shares at $100. A partial fill of 50 shares fires an `E` message with `executed_shares=50`. The buggy code removes all 500 shares from the book instead of just 50.

**Impact**: Each partial fill incorrectly removes the entire order. Subsequent fills on the same order try to remove shares that are already gone, resulting in negative share counts that corrupt the `active_prices` tracking.

---

### Bug 3: E/C/X/D Messages After Replacement Use Stale Price/Shares

**Location**: `get_messages()` — merge logic for E, C, X, D messages

**What happened**: The merge for E, C, X, D messages used the **original A/F order's** price and shares, even if the order had been replaced by a `U` message with different price/shares:

```python
# OLD (buggy)
for m in messages[2: -3]:  # E, C, X, D
    data[m] = data[m].merge(orders, how='left')  # orders only has A/F entries
```

**Example**: Order A at $100 × 500 shares is replaced via `U` to $105 × 300 shares. When an `E` execution fires on the replaced order, the code still references `price=$100, shares=500` from the original A message.

**Impact**: Shares are removed from the wrong price level, leaving phantom shares at the old price and creating a deficit at the new price.

---

### How These Bugs Produce the Inversion

The three bugs compound:

1. **Phantom orders accumulate** from unfixed chained replacements (Bug 1)
2. **Over-removal** from executions wipes too many shares at correct price levels (Bug 2)
3. **Stale-price removals** subtract shares from wrong levels (Bug 3)

The cumulative effect corrupts the `current_orders` dictionary. When `prep_order_book_minute_data` selects:

- `nlargest(price)` for buy → picks from corrupted high-price phantom bids
- `nsmallest(price)` for sell → picks from corrupted low-price phantom asks

This produces the observed inversion where buy prices appear higher than sell prices.

---

## Fixes Applied

### Fix 1: Live Order Registry for Chained U Messages

Replaced the simple merge with a **live order registry** (dictionary) that tracks all order reference numbers, including those created by `U` messages:

```python
# NEW (fixed)
order_registry = {}

# Populate from A/F messages
for row in initial_orders.itertuples(index=False):
    order_registry[row.order_reference_number] = (
        row.buy_sell_indicator, row.shares, row.price
    )

# Process U messages in timestamp order to handle chains (A→B→C→...)
u_sorted = data['U'].sort_values('timestamp')
for row in u_sorted.itertuples(index=False):
    orig_ref = row.original_order_reference_number
    new_ref = row.new_order_reference_number
    if orig_ref in order_registry:
        old_buysell, _, _ = order_registry[orig_ref]
        order_registry[new_ref] = (old_buysell, row.shares, row.price)
```

All subsequent lookups (E, C, X, D, U) use this registry instead of the simple merge, ensuring chained replacements are properly resolved.

### Fix 2: Use `executed_shares` for E and C Messages

```python
# NEW (fixed)
elif message.type in ['E', 'C']:
    if not np.isnan(message.price):
        price = int(message.price)
        shares = -int(message.executed_shares)  # ← only the filled portion
```

Both `E` and `C` are execution message types that carry `executed_shares`. This correctly removes only the executed portion from the order book.

### Fix 3: Registry-Based Lookups for Current Price/Shares

All E, C, X, D messages now get their `buy_sell_indicator`, `shares`, and `price` from the live registry (which reflects post-replacement state), not from the stale A/F-only lookup table.

Additionally, a NaN guard was added for D and X messages to handle edge cases:

```python
# NEW (fixed)
else:  # X, D
    if not np.isnan(message.price) and not np.isnan(message.shares):
        price = int(message.price)
        shares = -int(message.shares)
```

---

## Files Modified

| File                                  | Changes                                 |
| ------------------------------------- | --------------------------------------- |
| `02_rebuild_nasdaq_order_book.py`   | `get_messages()` rewritten + loop fix |
| `02_rebuild_nasdaq_order_book_2.py` | `get_messages()` rewritten + loop fix |
| `02_rebuild_nasdaq_order_book_3.py` | `get_messages()` rewritten + loop fix |

---

## Verification Steps

1. **Delete the HDF5 cache** for the target stock:

   ```bash
   rm data/10302019.NASDAQ_ITCH50_AMZN_order_book.h5
   ```
2. **Re-run the script** to rebuild the order book from scratch.
3. **Validate** using `plot_order_book_profile`:

   - Buy (bid) prices should be **at or below** the trade price
   - Sell (ask) prices should be **at or above** the trade price
   - `best_bid < best_ask` at every timestamp
4. **Check the full-day scatter plot**: Blue (buy) dots should appear **below** the black price line, and red (sell) dots should appear **above** it.

---

## ITCH Message Type Reference

| Type | Name              | Has `buy_sell_indicator` |         Has `shares`         | Has `executed_shares` | Has `printable` |
| ---- | ----------------- | :------------------------: | :----------------------------: | :---------------------: | :---------------: |
| A    | Add Order         |             ✅             |               ✅               |           —           |        —        |
| F    | Add Order (MPID)  |             ✅             |               ✅               |           —           |        —        |
| U    | Replace Order     |       — (inherited)       |            ✅ (new)            |           —           |        —        |
| E    | Order Executed    |      — (from order)      |               —               |           ✅           |        —        |
| C    | Executed w/ Price |      — (from order)      |               —               |           ✅           |        ✅        |
| X    | Order Cancel      |      — (from order)      | — (uses `cancelled_shares`) |           —           |        —        |
| D    | Order Delete      |      — (from order)      |      — (uses full order)      |           —           |        —        |
| P    | Trade (Non-Cross) |  ✅ (always B after 2014)  |               ✅               |           —           |        —        |
| Q    | Cross Trade       |             —             |               ✅               |           —           |        —        |
