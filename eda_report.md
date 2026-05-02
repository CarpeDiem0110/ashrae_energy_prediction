# EDA Review Insights

## Corrections To Previous Interpretation

1. Zero meter readings are  9% of the dataset.

   - Total training rows: `20,216,100`
   - Zero meter readings: `1,873,976`
   - Overall zero-reading rate: `9.27%`

   Zero readings are important, but they are not the dominant class. The real issue is that the zero rate is uneven by meter type:

   | Meter Type | Total Rows | Zero Rows | Zero Rate |
   |---|---:|---:|---:|
   | Electricity | 12,060,910 | 530,169 | 4.40% |
   | Chilled Water | 4,182,440 | 656,504 | 15.70% |
   | Steam | 2,708,713 | 346,960 | 12.81% |
   | Hot Water | 1,264,037 | 340,343 | 26.93% |

   ML interpretation: zeros should be analyzed as a data quality / operational pattern, especially by `meter`, `site_id`, `building_id`, and time. But this is not a "mostly zero target" problem.

2. The target is right-skewed, but zeros are not the only reason.

   The raw `meter_reading` distribution is highly skewed because:

   - meter types have very different physical scales
   - steam readings are much larger on average
   - extreme readings exist in the upper tail
   - buildings have very different sizes and use cases

   `log1p(meter_reading)` is the right modeling target for a first baseline because it reduces the impact of extreme values and aligns with RMSLE-style evaluation.



5. `building_id = 1099` should be treated as a hypothesis, not a confirmed insight from the current notebook output.

   The current notebook output confirms that extreme readings are concentrated by meter type, especially steam. It does not currently show a reliable table proving that `building_id = 1099` is responsible for many of the largest values. Add a top-outlier table by `building_id`, `site_id`, `meter`, and `timestamp` before keeping this as a final insight.

## Confirmed Findings

1. Meter type is one of the strongest drivers.

   Average meter reading by meter type:

   | Meter Type | Mean Meter Reading |
   |---|---:|
   | Electricity | 170.83 |
   | Chilled Water | 633.36 |
   | Steam | 13,882.19 |
   | Hot Water | 385.87 |

   Steam has a much larger scale than the other meter types. Direct raw comparisons across meter types are misleading.

2. Extreme values are heavily concentrated in steam.

   The top `0.1%` threshold is `38,671.9`, producing `20,220` rows.

   Distribution of top `0.1%` rows:

   | Meter Type | Top 0.1% Rows |
   |---|---:|
   | Steam | 17,326 |
   | Hot Water | 1,695 |
   | Chilled Water | 1,196 |
   | Electricity | 3 |

   ML interpretation: outlier handling should be meter-aware. A global cap or global outlier rule may damage valid steam behavior.

3. Weather effects are meter-specific.

   Correlation with `log_meter_reading`:

   | Meter | Air Temperature | Dew Temperature | Interpretation |
   |---:|---:|---:|---|
   | 0 | -0.004 | -0.040 | Electricity has weak linear weather correlation overall |
   | 1 | 0.433 | 0.429 | Chilled water increases with warmer weather |
   | 2 | -0.407 | -0.391 | Steam decreases as weather gets warmer |
   | 3 | -0.418 | -0.221 | Hot water also decreases as weather gets warmer |

   This is physically reasonable: chilled water behaves like cooling demand, while steam and hot water behave like heating demand.

4. Weather data has useful variables but also serious missingness.

   After merging weather into training data:

   - `air_temperature`: 96,658 missing
   - `dew_temperature`: 100,140 missing
   - `cloud_coverage`: 8,825,365 missing
   - `precip_depth_1_hr`: 3,749,023 missing
   - `sea_level_pressure`: 1,231,669 missing
   - `wind_direction`: 1,449,048 missing
   - `wind_speed`: 143,676 missing

   ML interpretation: temperature features are immediately useful. Sparse weather features need imputation and missingness indicators before modeling.

5. Building metadata has substantial missingness.

   After merging with training rows:

   - `year_built`: 12,127,645 missing rows
   - `floor_count`: 16,709,167 missing rows

   At the building table level:

   - `year_built` missing for 774 of 1,449 buildings, around 53.4%
   - `floor_count` missing for 1,094 of 1,449 buildings, around 75.5%

   

6. Square feet is useful, but should be transformed.

   The log-scale plot shows a positive relationship between `square_feet` and average meter reading. Larger buildings generally consume more, but the spread is large.

   ML interpretation:

   - use `log1p(square_feet)`
   - consider `meter_reading / square_feet` for EDA
   - avoid interpreting raw consumption as efficiency without normalizing by building size

7. Train/test feature completeness is well aligned.

   The parity check on the final modeling frames shows that train and test have very similar feature completeness levels.

   Most complete features:

   - `building_id`
   - `meter`
   - `timestamp`
   - `site_id`
   - `primary_use`
   - `square_feet`
   - `hour`
   - `weekday`
   - `month`
   - `is_weekend`
   - `air_temperature`
   - `dew_temperature`
   - `wind_speed`

   Weakest coverage features:

   - `floor_count`
   - `year_built`
   - `cloud_coverage`
   - `precip_depth_1_hr`

   ML interpretation: there is no major train/test missingness shift blocking baseline modeling. The issue is feature quality, not parity.

## Notebook Issues To Fix Later

1. In the dew-temperature section, this line should use `train_weather`, not `train`:

   ```python
   train_weather["log_meter_reading"] = np.log1p(train["meter_reading"])
   ```

   Better:

   ```python
   train_weather["log_meter_reading"] = np.log1p(train_weather["meter_reading"])
   ```


3. Add a direct outlier table before making building-level claims:

   ```python
   train.nlargest(20, "meter_reading")[
       ["building_id", "site_id", "primary_use", "meter", "timestamp", "meter_reading", "square_feet"]
   ]
   ```




