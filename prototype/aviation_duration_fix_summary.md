# Aviation Transport Duration Fix

## Summary of Changes

This update addresses unrealistic travel times for aviation transportation in the logistics route planner application. Previously, aviation transport was using the same duration calculation as ground transport, resulting in unrealistically long travel times for air shipments.

## Implemented Fixes

1. **Enhanced Maps Client**: 
   - Added separate `air_duration` field in route segments based on aviation speed (750 km/h)
   - Fixed indentation issues in the `extract_route_data` method
   - Properly documented the dual-duration approach in comments

2. **Carbon Calculator**: 
   - Updated to check for aviation vehicle types (`cargo_plane_*` and `sustainable_aviation`)
   - Added logic to use `air_duration` instead of standard duration for aviation vehicles
   - Fixed both batch prediction and individual segment calculation paths

3. **Validation Testing**:
   - Created a test script that verifies correct duration calculation for different vehicle types
   - Confirmed aviation vehicles now show realistic speeds (~750 km/h) compared to ground vehicles (~60 km/h)

## Technical Details

- Air transport duration is calculated using a speed of 750 km/h (compared to 60 km/h for ground)
- Vehicle type check uses ID prefix: `startswith("cargo_plane")` or `== "sustainable_aviation"`
- The standard duration field is maintained for backward compatibility

## Results

Before fix:
- All vehicles showed same duration regardless of type
- Aviation transport had unrealistically long travel times

After fix:
- Aviation transport shows appropriate travel times (~6x faster than ground)
- Ground transport durations remain unchanged
- Optimizer can now correctly compare time/cost tradeoffs between transport modes
