    def _setup_matplotlib(self):
        """Set up matplotlib with proper configuration."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # **FIXED: Configure matplotlib to handle categorical data properly**
            # Suppress the categorical units warning
            import warnings
            warnings.filterwarnings('ignore', message='Using categorical units to plot a list of strings')
            warnings.filterwarnings('ignore', category=matplotlib.category.UnicodeWarning)
            
            # Set default style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # **ENHANCED: Configure proper numeric handling for string data**
            # Set matplotlib to handle mixed data types gracefully
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.labelsize'] = 12
            plt.rcParams['xtick.labelsize'] = 10
            plt.rcParams['ytick.labelsize'] = 10
            plt.rcParams['legend.fontsize'] = 10
            
            # **FIXED: Configure categorical data handling**
            plt.rcParams['axes.formatter.use_mathtext'] = True
            
            self.plt = plt
            self.sns = sns
            
            logger.info("✅ Matplotlib configured with categorical data fix")
            
        except Exception as e:
            logger.error(f"Error setting up matplotlib: {e}")
            raise
    
    def _ensure_numeric_data(self, data, columns):
        """Ensure data columns are properly numeric for plotting."""
        try:
            processed_data = data.copy()
            
            for col in columns:
                if col in processed_data.columns:
                    # **FIXED: Convert string numbers to float, keep strings as categorical**
                    if processed_data[col].dtype == 'object':
                        # Try to convert to numeric if possible
                        numeric_conversion = pd.to_numeric(processed_data[col], errors='coerce')
                        
                        # If more than 50% can be converted to numeric, use numeric
                        if numeric_conversion.notna().sum() / len(processed_data) > 0.5:
                            processed_data[col] = numeric_conversion
                            logger.debug(f"Converted {col} to numeric for plotting")
                        else:
                            # Keep as categorical but ensure consistent ordering
                            processed_data[col] = processed_data[col].astype('category')
                            logger.debug(f"Kept {col} as categorical for plotting")
            
            return processed_data
            
        except Exception as e:
            logger.warning(f"Error ensuring numeric data: {e}")
            return data
    
    def _plot_with_numeric_handling(self, plot_func, data, x_col, y_col, **kwargs):
        """Plot data with proper numeric handling to avoid categorical warnings."""
        try:
            # **FIXED: Ensure proper data types before plotting**
            plot_data = self._ensure_numeric_data(data, [x_col, y_col])
            
            # **ENHANCED: Handle ZIP codes specifically (common source of warnings)**
            if 'zip' in x_col.lower():
                # Convert ZIP codes to strings and limit display
                plot_data[x_col] = plot_data[x_col].astype(str)
                
                # If too many ZIP codes, show only top N
                if len(plot_data[x_col].unique()) > 20:
                    top_zips = plot_data.nlargest(20, y_col)[x_col].unique()
                    plot_data = plot_data[plot_data[x_col].isin(top_zips)]
                    logger.info(f"Limited to top 20 ZIP codes for cleaner visualization")
            
            # **FIXED: Use appropriate plot function with categorical data handling**
            if plot_data[x_col].dtype.name == 'category' or plot_data[x_col].dtype == 'object':
                # For categorical x-axis, ensure proper ordering
                if 'zip' in x_col.lower():
                    # Sort ZIP codes numerically
                    plot_data = plot_data.sort_values(by=x_col)
                
                # Use bar plot for categorical data
                result = plot_func(data=plot_data, x=x_col, y=y_col, **kwargs)
                
                # Rotate x-axis labels for readability
                self.plt.xticks(rotation=45, ha='right')
            else:
                # Use line/scatter plot for numeric data
                result = plot_func(data=plot_data, x=x_col, y=y_col, **kwargs)
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in numeric plotting: {e}")
            # Fallback to original plot function
            return plot_func(data=data, x=x_col, y=y_col, **kwargs) 