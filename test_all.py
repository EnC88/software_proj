import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import logging
from src.data_processing import process_data
from src.vectorizer import Vectorizer
import time
import shutil
import threading
from sqlalchemy import text
from src.database.upgrade_db import Database
from src.cache.query_cache import QueryCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestUpgradeSystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.test_dir, 'input.csv')
        self.output_file = os.path.join(self.test_dir, 'output.csv')
        
        # Create test data
        self.test_data = pd.DataFrame({
            'OBJECTNAME': ['SoftwareA', 'SoftwareB', 'SoftwareC', 'SoftwareA'],
            'OLDVALUE': ['1.0.0', '1.0.0', '2.0.0', '2.0.0'],
            'NEWVALUE': ['2.0.0', '1.5.0', '3.0.0', '1.0.0'],
            'VERUMCREATEDBY': ['user1', 'user2', 'user3', 'user1'],
            'VERUMCREATEDDATE': ['2024-01-01', '2024-01-02', '2024-01-04', '2024-01-03'],
            'IS_ROLLBACK': [False, False, False, True],
            'IS_CLUSTERED_UPGRADE': [False, False, False, False]
        })
        self.test_data.to_csv(self.input_file, index=False)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_data_processing(self):
        """Test data processing functionality."""
        logger.info("Testing data processing...")
        
        # Process test data
        start_time = time.time()
        process_data(self.input_file, self.output_file)
        processing_time = time.time() - start_time
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        # Verify output file exists
        self.assertTrue(os.path.exists(self.output_file), "Output file should be created")
        
        # Read processed data
        processed_df = pd.read_csv(self.output_file)
        
        # Verify data integrity
        self.assertEqual(len(processed_df), 4, "Should process all records")
        
        # Check for clustered upgrades
        if 'IS_CLUSTERED_UPGRADE' in processed_df.columns:
            clustered_count = processed_df['IS_CLUSTERED_UPGRADE'].sum()
            logger.info(f"Number of clustered upgrades: {clustered_count}")
            self.assertEqual(clustered_count, 0, "Should have no clustered upgrades")
        
        # Check for rollbacks
        if 'IS_ROLLBACK' in processed_df.columns:
            rollback_count = processed_df['IS_ROLLBACK'].sum()
            logger.info(f"Number of rollbacks: {rollback_count}")
            self.assertEqual(rollback_count, 1, "Should have one rollback")
    
    def test_vectorizer(self):
        """Test vectorizer functionality with caching."""
        logger.info("Testing vectorizer...")
        
        # Initialize vectorizer with caching
        vectorizer = Vectorizer(use_cache=True)
        
        # Clear cache to ensure clean state
        vectorizer.query_cache.clear()
        
        # Process test data
        process_data(self.input_file, self.output_file)
        
        # Read processed data
        processed_df = pd.read_csv(self.output_file)
        
        # Create CHUNK_TEXT column by combining relevant information
        processed_df['CHUNK_TEXT'] = processed_df.apply(
            lambda row: (
                f"Software: {row['OBJECTNAME']}, "
                f"Version Change: {row['OLDVALUE']} to {row['NEWVALUE']}, "
                f"User: {row['VERUMCREATEDBY']}, "
                f"Date: {row['VERUMCREATEDDATE']}"
            ),
            axis=1
        )
        
        # Add additional context based on change type
        if 'IS_ROLLBACK' in processed_df.columns:
            processed_df['CHUNK_TEXT'] = processed_df.apply(
                lambda row: (
                    f"{row['CHUNK_TEXT']}, "
                    f"Type: {'Rollback' if row['IS_ROLLBACK'] else 'Upgrade'}"
                ),
                axis=1
            )
        
        if 'IS_CLUSTERED_UPGRADE' in processed_df.columns:
            processed_df['CHUNK_TEXT'] = processed_df.apply(
                lambda row: (
                    f"{row['CHUNK_TEXT']}, "
                    f"Clustered: {'Yes' if row['IS_CLUSTERED_UPGRADE'] else 'No'}"
                ),
                axis=1
            )
        
        # Set the processed data
        vectorizer.chunked_df = processed_df
        
        # Generate vectors
        start_time = time.time()
        vectorizer.vectorize()
        vectorization_time = time.time() - start_time
        
        logger.info(f"Vectorization completed in {vectorization_time:.2f} seconds")
        
        # Test queries
        test_queries = [
            "How to upgrade SoftwareA?",
            "What are common issues with SoftwareB?",
            "How to handle rollbacks?"
        ]
        
        # Track cache hits and misses
        cache_hits = 0
        cache_misses = 0
        
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            
            # First query (should miss cache)
            start_time = time.time()
            results = vectorizer.query_upgrades(query)
            first_query_time = time.time() - start_time
            
            # Second query (should hit cache)
            start_time = time.time()
            cached_results = vectorizer.query_upgrades(query)
            second_query_time = time.time() - start_time
            
            # Verify results
            self.assertEqual(results, cached_results, "Cached results should match original results")
            
            # Log timing and cache status
            logger.info(f"First query time: {first_query_time:.2f} seconds")
            logger.info(f"Second query time: {second_query_time:.2f} seconds")
            
            # Check if this was a cache hit based on timing
            is_cache_hit = second_query_time < first_query_time
            logger.info(f"Cache hit: {is_cache_hit}")
            
            # Update cache statistics
            # First query is always a miss, second query is a hit
            cache_misses += 1
            cache_hits += 1
            
            # Generate and display answer
            answer = vectorizer.generate_answer(query, results)
            logger.info(f"Answer:\n{answer}")
        
        # Display cache statistics
        stats = vectorizer.get_statistics()
        logger.info("\nCache Statistics:")
        if 'cache' in stats:
            cache_stats = stats['cache']
            logger.info(f"Cache hits: {cache_stats.get('hits', 0)}")
            logger.info(f"Cache misses: {cache_stats.get('misses', 0)}")
            logger.info(f"Cache size: {cache_stats.get('size', 0)}")
            logger.info(f"Hit rate: {cache_stats.get('hit_rate', 0)}%")
            logger.info(f"Cache uptime: {cache_stats.get('uptime', 0)} hours")
            
            # Verify cache statistics match our tracking
            self.assertEqual(cache_stats.get('hits', 0), cache_hits, 
                           "Cache hits should match tracked hits")
            self.assertEqual(cache_stats.get('misses', 0), cache_misses,
                           "Cache misses should match tracked misses")
        else:
            logger.info("No cache statistics available")
        
        logger.info(f"\nVectorizer test completed in {time.time() - start_time:.2f} seconds")
    
    def test_edge_cases(self):
        """Test edge cases for the vectorizer and cache system."""
        logger.info("\nTesting edge cases...")
        
        # Initialize database and vectorizer
        db = Database()
        vectorizer = Vectorizer(db)
        
        # Process test data
        process_data(self.input_file, self.output_file)
        
        # Load data into vectorizer
        processed_df = pd.read_csv(self.output_file)
        
        # Create CHUNK_TEXT column
        processed_df['CHUNK_TEXT'] = processed_df.apply(
            lambda row: (
                f"Software: {row['OBJECTNAME']}, "
                f"Version Change: {row['OLDVALUE']} to {row['NEWVALUE']}, "
                f"User: {row['VERUMCREATEDBY']}, "
                f"Date: {row['VERUMCREATEDDATE']}"
            ),
            axis=1
        )
        
        # Add additional context based on change type
        if 'IS_ROLLBACK' in processed_df.columns:
            processed_df['CHUNK_TEXT'] = processed_df.apply(
                lambda row: (
                    f"{row['CHUNK_TEXT']}, "
                    f"Type: {'Rollback' if row['IS_ROLLBACK'] else 'Upgrade'}"
                ),
                axis=1
            )
        
        if 'IS_CLUSTERED_UPGRADE' in processed_df.columns:
            processed_df['CHUNK_TEXT'] = processed_df.apply(
                lambda row: (
                    f"{row['CHUNK_TEXT']}, "
                    f"Clustered: {'Yes' if row['IS_CLUSTERED_UPGRADE'] else 'No'}"
                ),
                axis=1
            )
        
        # Set the processed data
        vectorizer.chunked_df = processed_df
        
        # Generate vectors
        vectorizer.vectorize()
        
        # Test 1: Empty query
        logger.info("\nTest 1: Empty query")
        results = vectorizer.query_upgrades("")
        self.assertIsNotNone(results, "Results should not be None for empty query")
        self.assertIsInstance(results, list, "Results should be a list")
        self.assertEqual(len(results), 0, "Empty query should return empty results")
        
        # Test 2: Very long query
        logger.info("\nTest 2: Very long query")
        long_query = "How to upgrade SoftwareA " * 100
        results = vectorizer.query_upgrades(long_query)
        self.assertIsNotNone(results, "Results should not be None for long query")
        self.assertIsInstance(results, list, "Results should be a list")
        
        # Test 3: Special characters in query
        logger.info("\nTest 3: Special characters in query")
        special_query = "How to upgrade SoftwareA? #@$%^&*()_+"
        results = vectorizer.query_upgrades(special_query)
        self.assertIsNotNone(results, "Results should not be None for special characters")
        self.assertIsInstance(results, list, "Results should be a list")
        
        # Test 4: Cache TTL expiration
        logger.info("\nTest 4: Cache TTL expiration")
        # Set short TTL
        vectorizer.query_cache.ttl = 1
        # Add query to cache
        results1 = vectorizer.query_upgrades("Test TTL query")
        # Wait for TTL to expire
        time.sleep(2)
        # Query should miss cache
        results2 = vectorizer.query_upgrades("Test TTL query")
        self.assertIsNotNone(results1, "First results should not be None")
        self.assertIsNotNone(results2, "Second results should not be None")
        
        # Test 5: Cache size limit
        logger.info("\nTest 5: Cache size limit")
        vectorizer.query_cache.clear()
        vectorizer.query_cache.max_size = 2
        
        # Add queries to fill cache
        vectorizer.query_upgrades("Query1")
        vectorizer.query_upgrades("Query2")
        vectorizer.query_upgrades("Query3")
        
        # Verify cache size
        self.assertLessEqual(len(vectorizer.query_cache.cache), 2, "Cache size should not exceed limit")
        
        # Test 6: Invalid data handling
        logger.info("\nTest 6: Invalid data handling")
        # Reset vectorizer state
        vectorizer.chunked_df = None
        vectorizer.vectors = None
        vectorizer.query_cache.clear()
        
        # Reset database state
        if vectorizer.use_database:
            with vectorizer.db.engine.connect() as conn:
                conn.execute(text("DELETE FROM upgrades"))
                conn.execute(text("DELETE FROM upgrades_fts"))
                conn.commit()
        
        # Test with invalid DataFrame
        vectorizer.chunked_df = pd.DataFrame({
            'CHUNK_TEXT': ['Invalid text'],
            'OBJECTNAME': ['SoftwareX'],
            'OLDVALUE': ['invalid'],
            'NEWVALUE': ['invalid']
        })
        # Force vectorizer to reinitialize
        vectorizer.vectors = None
        results = vectorizer.query_upgrades("Test invalid data")
        self.assertIsNotNone(results, "Results should not be None for invalid data")
        self.assertIsInstance(results, list, "Results should be a list")
        self.assertEqual(len(results), 0, "Invalid data should return empty results")
        
        # Test 7: Concurrent access
        logger.info("\nTest 7: Concurrent access")
        def query_worker():
            vectorizer.query_upgrades("Concurrent query")
        
        threads = [threading.Thread(target=query_worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Test 8: Cache persistence
        logger.info("\nTest 8: Cache persistence")
        query = "Test persistence"
        results = vectorizer.query_upgrades(query)
        
        # Create new vectorizer instance
        new_vectorizer = Vectorizer(use_cache=True)
        cached_results = new_vectorizer.query_upgrades(query)
        self.assertEqual(results, cached_results, "Cache should persist between instances")
        
        # Test 9: Error handling
        logger.info("\nTest 9: Error handling")
        # Test with None values
        vectorizer.chunked_df = None
        results = vectorizer.query_upgrades("Test error handling")
        self.assertEqual(len(results), 0, "Should handle None values gracefully")
        
        # Test 10: Cache statistics accuracy
        logger.info("\nTest 10: Cache statistics accuracy")
        vectorizer.query_cache.clear()
        query = "Test statistics"
        vectorizer.query_upgrades(query)  # Miss
        vectorizer.query_upgrades(query)  # Hit
        stats = vectorizer.get_statistics()
        if 'cache' in stats:
            self.assertEqual(stats['cache']['hits'], 1, "Cache hits should be accurate")
            self.assertEqual(stats['cache']['misses'], 1, "Cache misses should be accurate")
            self.assertEqual(stats['cache']['hit_rate'], 50.0, "Hit rate should be accurate")
        
        logger.info("\nEdge case tests completed")

    def test_error_handling(self):
        """Test error handling for various edge cases."""
        logger.info("\nTesting error handling...")
        
        # Initialize database and vectorizer
        db = Database()
        vectorizer = Vectorizer(db)
        
        # Test 1: Invalid query types
        logger.info("\nTest 1: Invalid query types")
        invalid_queries = [
            None,
            123,
            ["list", "of", "strings"],
            {"key": "value"},
            True,
            b"bytes query"
        ]
        for query in invalid_queries:
            results = vectorizer.query_upgrades(query)
            self.assertIsNotNone(results, f"Results should not be None for query type {type(query)}")
            self.assertIsInstance(results, list, f"Results should be a list for query type {type(query)}")
            self.assertEqual(len(results), 0, f"Invalid query type {type(query)} should return empty results")
        
        # Test 2: Malformed DataFrame
        logger.info("\nTest 2: Malformed DataFrame")
        malformed_dfs = [
            pd.DataFrame(),  # Empty DataFrame
            pd.DataFrame({'WRONG_COLUMN': ['value']}),  # Missing required columns
            pd.DataFrame({
                'CHUNK_TEXT': [None],
                'OBJECTNAME': ['SoftwareX'],
                'OLDVALUE': ['1.0'],
                'NEWVALUE': ['2.0']
            }),  # None values in required columns
            pd.DataFrame({
                'CHUNK_TEXT': [123],  # Wrong type
                'OBJECTNAME': ['SoftwareX'],
                'OLDVALUE': ['1.0'],
                'NEWVALUE': ['2.0']
            })
        ]
        
        for df in malformed_dfs:
            vectorizer.chunked_df = df
            vectorizer.vectors = None  # Force reinitialization
            results = vectorizer.query_upgrades("Test query")
            self.assertIsNotNone(results, f"Results should not be None for malformed DataFrame")
            self.assertIsInstance(results, list, "Results should be a list")
            self.assertEqual(len(results), 0, "Malformed DataFrame should return empty results")
        
        # Test 3: Database connection errors
        logger.info("\nTest 3: Database connection errors")
        # Temporarily break database connection
        original_engine = vectorizer.db.engine
        vectorizer.db.engine = None
        
        results = vectorizer.query_upgrades("Test query")
        self.assertIsNotNone(results, "Results should not be None for database error")
        self.assertIsInstance(results, list, "Results should be a list")
        self.assertEqual(len(results), 0, "Database error should return empty results")
        
        # Restore database connection
        vectorizer.db.engine = original_engine
        
        # Test 4: Model loading errors
        logger.info("\nTest 4: Model loading errors")
        # Temporarily break model
        original_model = vectorizer.model
        vectorizer.model = None
        
        results = vectorizer.query_upgrades("Test query")
        self.assertIsNotNone(results, "Results should not be None for model error")
        self.assertIsInstance(results, list, "Results should be a list")
        self.assertEqual(len(results), 0, "Model error should return empty results")
        
        # Restore model
        vectorizer.model = original_model
        
        # Test 5: Cache errors
        logger.info("\nTest 5: Cache errors")
        # Temporarily break cache
        original_cache = vectorizer.query_cache
        vectorizer.query_cache = None
        
        results = vectorizer.query_upgrades("Test query")
        self.assertIsNotNone(results, "Results should not be None for cache error")
        self.assertIsInstance(results, list, "Results should be a list")
        
        # Restore cache
        vectorizer.query_cache = original_cache
        
        # Test 6: Vectorization errors
        logger.info("\nTest 6: Vectorization errors")
        # Create DataFrame with invalid text that might cause vectorization issues
        vectorizer.chunked_df = pd.DataFrame({
            'CHUNK_TEXT': ['\x00' * 1000],  # Null bytes
            'OBJECTNAME': ['SoftwareX'],
            'OLDVALUE': ['1.0'],
            'NEWVALUE': ['2.0']
        })
        vectorizer.vectors = None  # Force reinitialization
        
        results = vectorizer.query_upgrades("Test query")
        self.assertIsNotNone(results, "Results should not be None for vectorization error")
        self.assertIsInstance(results, list, "Results should be a list")
        self.assertEqual(len(results), 0, "Vectorization error should return empty results")
        
        logger.info("Error handling tests completed")

    def test_performance_and_resources(self):
        """Test performance and resource usage under various conditions."""
        logger.info("\nTesting performance and resources...")
        
        # Initialize database and vectorizer
        db = Database()
        vectorizer = Vectorizer(db)
        
        # Test 1: Large dataset handling
        logger.info("\nTest 1: Large dataset handling")
        # Create a large dataset
        large_df = pd.DataFrame({
            'CHUNK_TEXT': [f"Test text {i}" for i in range(1000)],
            'OBJECTNAME': ['SoftwareX'] * 1000,
            'OLDVALUE': ['1.0'] * 1000,
            'NEWVALUE': ['2.0'] * 1000
        })
        
        start_time = time.time()
        vectorizer.chunked_df = large_df
        vectorizer.vectorize()
        vectorization_time = time.time() - start_time
        
        logger.info(f"Vectorization time for 1000 records: {vectorization_time:.2f} seconds")
        self.assertLess(vectorization_time, 30.0, "Vectorization should complete within 30 seconds")
        
        # Test 2: Memory usage
        logger.info("\nTest 2: Memory usage")
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple queries
        for _ in range(10):
            vectorizer.query_upgrades("Test query")
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        logger.info(f"Memory increase: {memory_increase:.2f} MB")
        self.assertLess(memory_increase, 500, "Memory usage should not increase by more than 500MB")
        
        # Test 3: Query response time
        logger.info("\nTest 3: Query response time")
        query_times = []
        for _ in range(10):
            start_time = time.time()
            vectorizer.query_upgrades("Test query")
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        avg_query_time = sum(query_times) / len(query_times)
        logger.info(f"Average query time: {avg_query_time:.3f} seconds")
        self.assertLess(avg_query_time, 1.0, "Average query time should be less than 1 second")
        
        # Test 4: Cache performance
        logger.info("\nTest 4: Cache performance")
        # Clear cache
        vectorizer.query_cache.clear()
        
        # First query (cache miss)
        start_time = time.time()
        vectorizer.query_upgrades("Cache test query")
        cache_miss_time = time.time() - start_time
        
        # Second query (cache hit)
        start_time = time.time()
        vectorizer.query_upgrades("Cache test query")
        cache_hit_time = time.time() - start_time
        
        logger.info(f"Cache miss time: {cache_miss_time:.3f} seconds")
        logger.info(f"Cache hit time: {cache_hit_time:.3f} seconds")
        self.assertLess(cache_hit_time, cache_miss_time, "Cache hit should be faster than cache miss")
        
        # Test 5: Concurrent query performance
        logger.info("\nTest 5: Concurrent query performance")
        import concurrent.futures
        
        def run_query(query):
            return vectorizer.query_upgrades(query)
        
        queries = [f"Concurrent query {i}" for i in range(5)]
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(run_query, queries))
        
        concurrent_time = time.time() - start_time
        logger.info(f"Concurrent query time: {concurrent_time:.3f} seconds")
        self.assertLess(concurrent_time, 5.0, "Concurrent queries should complete within 5 seconds")
        
        # Test 6: Database connection pool
        logger.info("\nTest 6: Database connection pool")
        if hasattr(vectorizer.db.engine, 'pool'):
            pool_size = vectorizer.db.engine.pool.size()
            logger.info(f"Connection pool size: {pool_size}")
            self.assertGreater(pool_size, 0, "Connection pool should be initialized")
        
        logger.info("Performance and resource tests completed")

    def test_stress_and_metrics(self):
        """Test system performance under stress and collect detailed metrics."""
        logger.info("\nTesting stress conditions and collecting metrics...")
        
        # Initialize database and vectorizer
        db = Database()
        vectorizer = Vectorizer(db)
        
        # Test 1: Extreme dataset sizes
        logger.info("\nTest 1: Extreme dataset sizes")
        dataset_sizes = [100, 1000, 5000, 10000]
        metrics = {}
        
        for size in dataset_sizes:
            logger.info(f"\nTesting with {size} records...")
            # Create dataset
            large_df = pd.DataFrame({
                'CHUNK_TEXT': [f"Test text {i}" for i in range(size)],
                'OBJECTNAME': ['SoftwareX'] * size,
                'OLDVALUE': ['1.0'] * size,
                'NEWVALUE': ['2.0'] * size
            })
            
            # Measure vectorization
            start_time = time.time()
            vectorizer.chunked_df = large_df
            vectorizer.vectorize()
            vectorization_time = time.time() - start_time
            
            # Measure memory
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            # Measure query performance
            query_times = []
            for _ in range(5):
                start_time = time.time()
                vectorizer.query_upgrades("Test query")
                query_times.append(time.time() - start_time)
            
            avg_query_time = sum(query_times) / len(query_times)
            
            metrics[size] = {
                'vectorization_time': vectorization_time,
                'memory_usage': memory_usage,
                'avg_query_time': avg_query_time
            }
            
            logger.info(f"Size {size}:")
            logger.info(f"  Vectorization time: {vectorization_time:.2f}s")
            logger.info(f"  Memory usage: {memory_usage:.2f}MB")
            logger.info(f"  Average query time: {avg_query_time:.3f}s")
        
        # Test 2: Query complexity
        logger.info("\nTest 2: Query complexity")
        complex_queries = [
            "How to upgrade SoftwareA from version 1.0 to 2.0 with minimal downtime?",
            "What are the common issues when upgrading SoftwareB in a clustered environment?",
            "How to handle rollbacks for SoftwareC when the upgrade fails?",
            "What are the best practices for upgrading multiple software components simultaneously?",
            "How to verify compatibility between SoftwareA and SoftwareB after upgrades?"
        ]
        
        query_metrics = {}
        for query in complex_queries:
            # First query (cache miss)
            start_time = time.time()
            results = vectorizer.query_upgrades(query)
            first_query_time = time.time() - start_time
            
            # Second query (cache hit)
            start_time = time.time()
            results = vectorizer.query_upgrades(query)
            second_query_time = time.time() - start_time
            
            query_metrics[query] = {
                'first_query_time': first_query_time,
                'second_query_time': second_query_time,
                'result_count': len(results),
                'avg_similarity': sum(r['similarity'] for r in results) / len(results) if results else 0
            }
            
            logger.info(f"\nQuery: {query}")
            logger.info(f"  First query time: {first_query_time:.3f}s")
            logger.info(f"  Second query time: {second_query_time:.3f}s")
            logger.info(f"  Results: {len(results)}")
            logger.info(f"  Average similarity: {query_metrics[query]['avg_similarity']:.3f}")
        
        # Test 3: Concurrent operations
        logger.info("\nTest 3: Concurrent operations")
        import concurrent.futures
        
        def run_concurrent_operation(operation_type, index):
            if operation_type == 'query':
                return vectorizer.query_upgrades(f"Concurrent query {index}")
            elif operation_type == 'vectorize':
                # Create small dataset for vectorization
                df = pd.DataFrame({
                    'CHUNK_TEXT': [f"Test text {i}" for i in range(100)],
                    'OBJECTNAME': ['SoftwareX'] * 100,
                    'OLDVALUE': ['1.0'] * 100,
                    'NEWVALUE': ['2.0'] * 100
                })
                vectorizer.chunked_df = df
                return vectorizer.vectorize()
        
        # Test different combinations of concurrent operations
        operation_combinations = [
            (['query'] * 5),  # 5 concurrent queries
            (['vectorize'] * 3),  # 3 concurrent vectorizations
            (['query'] * 3 + ['vectorize'] * 2),  # Mixed operations
        ]
        
        for operations in operation_combinations:
            logger.info(f"\nTesting {len(operations)} concurrent operations: {operations}")
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(operations)) as executor:
                futures = [
                    executor.submit(run_concurrent_operation, op, i)
                    for i, op in enumerate(operations)
                ]
                results = [f.result() for f in futures]
            
            total_time = time.time() - start_time
            logger.info(f"Total time: {total_time:.2f}s")
        
        # Test 4: Resource cleanup
        logger.info("\nTest 4: Resource cleanup")
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory after cleanup
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Memory after cleanup: {memory_after:.2f}MB")
        
        # Test 5: Database connection management
        logger.info("\nTest 5: Database connection management")
        if hasattr(vectorizer.db.engine, 'pool'):
            pool = vectorizer.db.engine.pool
            logger.info(f"Initial pool size: {pool.size()}")
            logger.info(f"Checked out connections: {pool.checkedin()}")
            logger.info(f"Overflow: {pool.overflow()}")
            logger.info(f"Timeout: {pool.timeout()}")
        
        # Test 6: Cache efficiency
        logger.info("\nTest 6: Cache efficiency")
        cache_stats = vectorizer.query_cache.get_stats()
        logger.info("Cache statistics:")
        logger.info(f"  Hits: {cache_stats['hits']}")
        logger.info(f"  Misses: {cache_stats['misses']}")
        logger.info(f"  Size: {cache_stats['size']}")
        logger.info(f"  Hit rate: {cache_stats['hit_rate']:.1f}%")
        logger.info(f"  Uptime: {cache_stats['uptime']:.1f} hours")
        
        # Test 7: Error recovery
        logger.info("\nTest 7: Error recovery")
        # Simulate various error conditions
        error_conditions = [
            lambda: vectorizer.db.engine.dispose(),  # Database connection error
            lambda: setattr(vectorizer, 'model', None),  # Model error
            lambda: setattr(vectorizer, 'query_cache', None),  # Cache error
            lambda: setattr(vectorizer, 'chunked_df', None),  # Data error
        ]
        
        for i, error_condition in enumerate(error_conditions):
            logger.info(f"\nTesting error recovery {i+1}")
            # Apply error condition
            error_condition()
            
            # Attempt to recover
            start_time = time.time()
            results = vectorizer.query_upgrades("Test recovery")
            recovery_time = time.time() - start_time
            
            logger.info(f"Recovery time: {recovery_time:.3f}s")
            self.assertIsNotNone(results, "Results should not be None after error")
            self.assertIsInstance(results, list, "Results should be a list after error")
        
        logger.info("\nStress and metrics tests completed")

if __name__ == '__main__':
    unittest.main() 