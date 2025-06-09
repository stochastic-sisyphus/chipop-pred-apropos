import logging
import sys
import traceback

def run_pipeline():
    """
    Run the Chicago Housing Pipeline.
    
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    try:
        logger.info("Starting Chicago Housing Pipeline")
        
        # Initialize components
        collector = DataCollector()
        validator = SchemaValidator()
        processor = DataProcessor()
        analyzer = DataAnalyzer()
        
        # Track collection status
        collection_status = {
            'census': False,
            'economic': False,
            'vacancy': False,
            'business': False,
            'migration': False
        }
        
        # Collect data
        logger.info("Collecting data...")
        
        # Census data
        try:
            census_data = collector.collect_census_data()
            if census_data is not None and not census_data.empty:
                is_valid, errors = validator.validate_dataframe(census_data, 'census')
                if is_valid:
                    collection_status['census'] = True
                    logger.info("Successfully collected and validated Census data")
                else:
                    logger.error(f"Census data validation failed: {errors}")
            else:
                logger.error("Failed to collect Census data")
        except Exception as e:
            logger.error(f"Error collecting Census data: {str(e)}")
        
        # Economic data
        try:
            economic_data = collector.collect_economic_data()
            if economic_data is not None and not economic_data.empty:
                is_valid, errors = validator.validate_dataframe(economic_data, 'economic')
                if is_valid:
                    collection_status['economic'] = True
                    logger.info("Successfully collected and validated economic data")
                else:
                    logger.error(f"Economic data validation failed: {errors}")
            else:
                logger.error("Failed to collect economic data")
        except Exception as e:
            logger.error(f"Error collecting economic data: {str(e)}")
        
        # Vacancy data
        try:
            vacancy_data = collector.collect_vacancy_data()
            if vacancy_data is not None and not vacancy_data.empty:
                is_valid, errors = validator.validate_dataframe(vacancy_data, 'vacancy')
                if is_valid:
                    collection_status['vacancy'] = True
                    logger.info("Successfully collected and validated vacancy data")
                else:
                    logger.error(f"Vacancy data validation failed: {errors}")
            else:
                logger.error("Failed to collect vacancy data")
        except Exception as e:
            logger.error(f"Error collecting vacancy data: {str(e)}")
        
        # Business data
        try:
            business_data = collector.collect_business_licenses()
            if business_data is not None and not business_data.empty:
                is_valid, errors = validator.validate_dataframe(business_data, 'business')
                if is_valid:
                    collection_status['business'] = True
                    logger.info("Successfully collected and validated business data")
                else:
                    logger.error(f"Business data validation failed: {errors}")
            else:
                logger.error("Failed to collect business data")
        except Exception as e:
            logger.error(f"Error collecting business data: {str(e)}")
        
        # Migration data
        try:
            migration_data = collector.collect_migration_data()
            if migration_data is not None and not migration_data.empty:
                is_valid, errors = validator.validate_dataframe(migration_data, 'migration')
                if is_valid:
                    collection_status['migration'] = True
                    logger.info("Successfully collected and validated migration data")
                else:
                    logger.error(f"Migration data validation failed: {errors}")
            else:
                logger.error("Failed to collect migration data")
        except Exception as e:
            logger.error(f"Error collecting migration data: {str(e)}")
        
        # Process data if we have at least some valid data
        if any(collection_status.values()):
            logger.info("Processing collected data...")
            try:
                processed_data = processor.process_data()
                if processed_data is not None and not processed_data.empty:
                    logger.info("Successfully processed data")
                else:
                    logger.error("Failed to process data")
            except Exception as e:
                logger.error(f"Error processing data: {str(e)}")
        else:
            logger.error("No valid data collected, skipping processing")
            return False
        
        # Analyze data if processing was successful
        if processed_data is not None and not processed_data.empty:
            logger.info("Analyzing data...")
            try:
                analysis_results = analyzer.analyze_data(processed_data)
                if analysis_results:
                    logger.info("Successfully analyzed data")
                else:
                    logger.error("Failed to analyze data")
            except Exception as e:
                logger.error(f"Error analyzing data: {str(e)}")
        else:
            logger.error("No processed data available, skipping analysis")
            return False
        
        # Print collection status summary
        logger.info("\nData Collection Status Summary:")
        for data_type, status in collection_status.items():
            status_str = "✓" if status else "✗"
            logger.info(f"{data_type.title()} Data: {status_str}")
        
        # Determine overall success
        success_rate = sum(collection_status.values()) / len(collection_status)
        if success_rate >= 0.6:  # At least 60% of data types collected successfully
            logger.info(f"\nPipeline completed with {success_rate:.0%} success rate")
            return True
        else:
            logger.error(f"\nPipeline failed with only {success_rate:.0%} success rate")
            return False
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    success = run_pipeline()
    sys.exit(0 if success else 1) 