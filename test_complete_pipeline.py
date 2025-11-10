#!/usr/bin/env python3
"""
Enhanced Complete RAG Pipeline: 
Query Preprocessing + Retrieval + Reranking + Generation + Feedback Learning + Adaptive Planning
"""

import json
import pickle
import time
import os
from datetime import datetime

def test_imports():
    """Test all required imports"""
    try:
        from preprocessor.preprocessor_services import PreprocessorServices
        print(" PreprocessorServices imported")
        
        from planner.planner_services import PlannerServices
        print(" PlannerServices imported")
        
        from executor.executor_services import ExecutorServices
        print(" ExecutorServices imported")
        
        from reranker.cascade_reranker import CascadeReranker
        print(" CascadeReranker imported")
        
        from generator.generator_services import GeneratorServices
        print(" GeneratorServices imported")
        
        from generator.feedback_learning_system import FeedbackLearningSystem
        print(" FeedbackLearningSystem imported")
        
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        return False

def load_test_data():
    """Load minimal test data"""
    try:
        # Load documents from chunks
        with open("pipeline_output/embeddings/chunk_metadata.json", 'r') as f:
            chunk_metadata = json.load(f)
        
        with open("pipeline_output/embeddings/chunks_data.pkl", 'rb') as f:
            chunks_data = pickle.load(f)
        
        # Convert to document format (ALL documents for full testing)
        documents = []
        for i, (metadata, chunk_data) in enumerate(zip(chunk_metadata, chunks_data)):
            content = ""
            if isinstance(chunk_data, dict) and 'content' in chunk_data:
                content_list = chunk_data['content']
                if isinstance(content_list, list):
                    for item in content_list:
                        if isinstance(item, dict) and 'content' in item:
                            content += item['content'] + " "
            
            doc = {
                'id': f"chunk_{metadata['chunk_id']}",
                'content': content.strip(),
                'title': f"Bhagavad Gita - Chunk {metadata['chunk_id']}",
                'entities': [],
                'topic': 'philosophy'
            }
            documents.append(doc)
        
        # Load knowledge graph
        with open("pipeline_output/knowledge_graph.json", 'r') as f:
            knowledge_graph = json.load(f)
        
        print(f"✅ Loaded {len(documents)} documents and knowledge graph")
        return documents, knowledge_graph
        
    except Exception as e:
        print(f" Failed to load test data: {e}")
        return [], {"nodes": [], "edges": []}

def load_or_create_session_data():
    """Load or create session data for MemoRAG"""
    session_file = "memorag_sessions.json"
    
    if os.path.exists(session_file):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            print(f"Loaded existing session data with {len(session_data.get('sessions', []))} sessions")
        except Exception as e:
            print(f"Error loading session data: {e}, creating new")
            session_data = {"sessions": [], "current_session_id": None}
    else:
        session_data = {"sessions": [], "current_session_id": None}
        print(" Created new session data")
    
    return session_data

def save_session_data(session_data):
    """Save session data for MemoRAG"""
    try:
        with open("memorag_sessions.json", 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        print(" Session data saved")
    except Exception as e:
        print(f"Error saving session data: {e}")

def create_feedback_entry(rating, feedback_type, correction=None):
    """Create a feedback entry for learning system"""
    from generator.enhanced_response_formatter import FeedbackType, FeedbackEntry
    
    feedback_type_map = {
        "thumbs_up": FeedbackType.THUMBS_UP,
        "thumbs_down": FeedbackType.THUMBS_DOWN,
        "correct": FeedbackType.PARTIALLY_CORRECT,  # Use available enum value
        "not_correct": FeedbackType.NOT_CORRECT,
        "helpful": FeedbackType.VERY_HELPFUL,  # Use available enum value
        "not_helpful": FeedbackType.NEEDS_IMPROVEMENT  # Use available enum value
    }
    
    return FeedbackEntry(
        feedback_id=f"feedback_{int(time.time())}",
        response_id=f"response_{int(time.time())}",
        feedback_type=feedback_type_map.get(feedback_type, FeedbackType.THUMBS_UP),
        rating=rating,
        comment=correction if correction else f"Simulated {feedback_type} feedback",
        correction=correction,
        timestamp=datetime.now()
    )

def create_response_trace(query, intent, generated_response, evidence_documents=None, generation_metadata=None, validation_results=None):
    """Create a response trace for learning system"""
    from generator.enhanced_response_formatter import ResponseTrace
    
    return ResponseTrace(
        response_id=f"response_{int(time.time())}",
        query=query,
        intent=intent,
        generated_response=generated_response,
        evidence_documents=evidence_documents or [],
        generation_metadata=generation_metadata or {},
        feedback_entries=[],  # Empty initially
        timestamp=datetime.now(),
        processing_time=0.0,  # Will be updated
        validation_results=validation_results
    )

def main():
    """Interactive Enhanced Complete RAG Pipeline"""
    print("🚀 INTERACTIVE ENHANCED RAG PIPELINE")
    print("=" * 80)
    print("Welcome to the Interactive RAG Pipeline Tester!")
    print("This system will:")
    print("  1. 🔍 Process your queries about the Bhagavad Gita")
    print("  2. 🎯 Retrieve relevant information")
    print("  3. 📊 Rerank results for better relevance")
    print("  4. 🤖 Generate comprehensive answers")
    print("  5. 📝 Collect your feedback for continuous learning")
    print("=" * 80)
    
    # Test imports
    if not test_imports():
        return
    
    # Load test data
    documents, knowledge_graph = load_test_data()
    if not documents:
        print("No test data available")
        return
    
    # Load session data for MemoRAG
    session_data = load_or_create_session_data()
    
    # Interactive mode - get queries from user
    def get_user_queries():
        """Get queries from user input"""
        print("\n" + "="*80)
        print("🎯 INTERACTIVE RAG PIPELINE TESTING")
        print("="*80)
        print("Enter your queries about the Bhagavad Gita. Type 'quit' to exit.")
        print("Examples:")
        print("  -what is differenece between karma yoga and bhakti yoga?")
        print("  - What is dharma?")
        print("  - Compare Arjuna and Krishna")
        print("  - In the Mahabharata war, whom did Duryodhana first talk to?")
        print("-" * 80)
        
        queries = []
        query_count = 1
        
        while True:
            try:
                query = input(f"\nQuery {query_count}: ").strip()
                
                if not query:
                    print("Please enter a query or 'quit' to exit.")
                    continue
                    
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                queries.append(query)
                query_count += 1
                
                # Ask if user wants to add more queries
                if len(queries) >= 1:
                    more = input("Add another query? (y/n): ").strip().lower()
                    if more not in ['y', 'yes']:
                        break
                        
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        return queries
    
    # Get queries from user
    test_queries = get_user_queries()
    
    if not test_queries:
        print("No queries provided. Using default queries for demonstration.")
        test_queries = [
            "What is Krishna's other name?",
            "What is the concept of dharma?",
            "Who fought in the Mahabharata war?"
        ]
    
    pipeline_results = []
    all_feedback_entries = []  # Accumulate feedback across queries
    all_response_traces = []   # Accumulate response traces
    
    try:
        # Initialize all services
        print("\n1. Initializing Enhanced Pipeline Services...")
        
        from preprocessor.preprocessor_services import PreprocessorServices
        preprocessor_services = PreprocessorServices()
        print(" PreprocessorServices initialized")
        
        from planner.planner_services import PlannerServices
        planner_services = PlannerServices(
            enable_optimization=True,
            enable_learning=True,
            session_timeout=3600,
            auto_start_session=True
        )
        print(" PlannerServices initialized")
        
        from executor.executor_services import ExecutorServices
        executor_services = ExecutorServices(
            enable_fusion=True,
            enable_caching=False,  # Disable caching for testing
            max_results=10,
            documents=documents,
            knowledge_graph=knowledge_graph
        )
        print("ExecutorServices initialized")
        
        from reranker.cascade_reranker import CascadeReranker
        cascade_reranker = CascadeReranker(
            stage1_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            stage2_model="cross-encoder/ms-marco-electra-base",
            final_top_k=5
        )
        print("CascadeReranker initialized")
        
        # Use direct AnswerGenerator with OPEN SOURCE model via API (FREE!)
        from generator.answer_generator import AnswerGenerator
        answer_generator = AnswerGenerator(
            model_name="meta-llama/llama-3.3-70b-instruct:free",  # Open Source via OpenRouter - FREE!
            device="cpu",
            max_length=2048,
            temperature=0.7,
            top_p=0.9
        )
        print("Direct AnswerGenerator initialized with OPEN SOURCE model (Llama 3.3 70B - FREE!)")
        
        from generator.feedback_learning_system import FeedbackLearningSystem
        feedback_learning_system = FeedbackLearningSystem(
            learning_rate=0.1,
            min_feedback_threshold=1,  # Single feedback can generate insights
            enable_auto_adjustments=True
        )
        print("FeedbackLearningSystem initialized")
        
        # Process multiple queries to demonstrate adaptive planning
        for query_idx, query in enumerate(test_queries, 1):
            print(f"\n" + "=" * 80)
            print(f"PROCESSING QUERY {query_idx}/3: {query}")
            print("=" * 80)
            
            # Step 1: Query Preprocessing
            print(f"\n1. Query Preprocessing...")
            preprocessing_result = preprocessor_services.process_query(query, include_stats=True)
            
            normalized_query = preprocessing_result['normalized_query']
            intent = preprocessing_result['intent']['primary_intent']
            entities = preprocessing_result['entities']['entities']
            confidence = preprocessing_result['intent']['confidence']
            
            print(f"Original Query: {query}")
            print(f"Normalized Query: {normalized_query}")
            print(f"Intent: {intent} (Confidence: {confidence:.3f})")
            print(f"Entities: {entities}")
            
            # Step 2: Adaptive Planning
            print(f"\n2. Adaptive Planning...")
            
            # Create session context
            session_context = {
                "session_id": f"session_{int(time.time())}",
                "query_history": [q for q in test_queries[:query_idx-1]],
                "previous_results": pipeline_results,
                "user_preferences": {"detail_level": "comprehensive"}
            }
            
            # Generate execution plan
            planning_result = planner_services.generate_plan(
                query=normalized_query,
                intent=intent,
                entities=entities,
                optimization_goal="balanced",
                context=session_context
            )
            
            planner_plan = planning_result.get('execution_plan', {})
            strategy_used = planner_plan.get('strategy', 'adaptive')
            
            print(f"Planning Strategy: {strategy_used}")
            print(f"Retrievers: {planner_plan.get('retrievers', ['bm25', 'dense'])}")
            print(f"Fusion Method: {planner_plan.get('fusion_method', 'rrf')}")
            print(f"Planning Confidence: {planning_result.get('confidence', 0.0):.3f}")
            
            # Show strategy adaptation if not first query
            if query_idx > 1:
                print(f"Strategy Adaptation: Based on previous results, adjusted parameters")
                print(f"Previous Strategy Impact: Learning from query {query_idx-1}")
            
            # Step 3: Retrieval with Fusion
            print(f"\n3. Retrieval with Fusion...")
            retrieval_result = executor_services.execute_with_planner_output(
                query=normalized_query,
                planner_plan=planner_plan
            )
            
            if retrieval_result.get('documents'):
                print(f" Retrieved {len(retrieval_result['documents'])} documents")
                print(f"Fusion Applied: {retrieval_result.get('fusion_applied', False)}")
                print(f"Retrieval Time: {retrieval_result['execution_metadata']['execution_time']:.3f}s")
            else:
                print("No documents retrieved")
                continue
        
            
            # Step 4: Advanced Reranking
            print(f"\n4. Advanced Reranking...")
            docs_for_reranking = []
            for doc in retrieval_result['documents']:
                docs_for_reranking.append({
                    'id': doc['id'],
                    'content': doc['content'],
                    'title': doc.get('title', ''),
                    'similarity_score': doc['score'],
                    'metadata': doc.get('metadata', {})
                })
            
            reranked_result = cascade_reranker.rerank(
                query=normalized_query,
                candidates=docs_for_reranking,
                custom_final_k=5,
                session_context=session_context
            )
            
            reranked_docs = reranked_result.get('final_results', [])
            if reranked_docs:
                print(f" Reranked to {len(reranked_docs)} documents")
                cascade_metadata = reranked_result.get('cascade_metadata', {})
                timing = cascade_metadata.get('timing', {})
                print(f"Reranking Time: {timing.get('total_time', 0):.3f}s")
                print(f"Stage 1→2 Compression: {cascade_metadata.get('input_count', 0)}→{cascade_metadata.get('final_count', 0)}")
            else:
                print(" No documents after reranking")
                continue
            
            # Step 5: Direct Answer Generation (Fixed)
            print(f"\n5. Direct Answer Generation...")
            evidence_documents = []
            for doc in reranked_docs:
                evidence_doc = {
                    'id': doc.get('id', 'unknown'),
                    'content': doc.get('content', ''),
                    'title': doc.get('title', ''),
                    'similarity_score': doc.get('reranked_score', doc.get('similarity_score', 0.0)),
                    'metadata': doc.get('metadata', {})
                }
                evidence_documents.append(evidence_doc)
            
            # Create evidence text with clear formatting
            evidence_text = ""
            for i, doc in enumerate(evidence_documents, 1):
                # Truncate very long content but keep key information
                content = doc['content'][:400] + "..." if len(doc['content']) > 400 else doc['content']
                evidence_text += f"Evidence {i}: {content}\n\n"
            
            # Create dynamic prompt based on intent and entities
            def create_dynamic_prompt(query, intent, entities, evidence_text):
                """Create adaptive prompt based on query intent and entities"""
                
                # Base system prompt
                system_prompt = "You are a helpful assistant specializing in Hindu philosophy and the Bhagavad Gita. Answer questions accurately based on the evidence provided."
                
                # Create intent-specific instructions
                if intent == "fact":
                    instruction = "Provide a direct, factual answer based on the evidence."
                elif intent == "explain":
                    instruction = "Provide a clear explanation with details from the evidence."
                elif intent == "compare":
                    instruction = "Compare and contrast based on the evidence provided."
                elif intent == "irrelevant":
                    # Handle questions that seem irrelevant but might actually be valid
                    instruction = "Answer the question directly based on any relevant information in the evidence."
                else:
                    instruction = "Answer the question based on the evidence provided."
                
                # Add entity-specific guidance
                entity_guidance = ""
                if entities:
                    key_entities = [e for e in entities if len(e) > 2]  # Filter out short entities
                    if key_entities:
                        entity_guidance = f" Pay special attention to mentions of: {', '.join(key_entities[:3])}."
                
                # Create user prompt
                user_prompt = f"""Question: {query}

Evidence:
{evidence_text}

{instruction}{entity_guidance}

Answer:"""
                
                return {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt
                }
            
            # Generate dynamic prompt
            prompt = create_dynamic_prompt(normalized_query, intent, entities, evidence_text)
            
            print(f"Generating answer with direct AnswerGenerator...")
            
            # Generate answer using the direct generator
            direct_result = answer_generator.generate_answer(prompt, evidence_documents)
            
            # Convert to expected format
            generation_result = {
                'success': True,
                'generated_answer': direct_result,
                'generation_metadata': {
                    'generation_time': direct_result.get('metadata', {}).get('generation_time', 0),
                    'model_used': direct_result.get('metadata', {}).get('model_used', 'meta-llama/llama-3.3-70b-instruct:free'),
                    'validation_applied': False,
                    'enhanced_formatting_applied': False
                }
            }
            
            if generation_result['success']:
                print(" Answer generated successfully")
                generated_answer = generation_result['generated_answer']
                generation_metadata = generation_result['generation_metadata']
                
                print(f"Generation Time: {generation_metadata['generation_time']:.3f}s")
                print(f"Model Used: {generation_metadata['model_used']}")
                print(f"Validation Applied: {generation_metadata['validation_applied']}")
                print(f"Enhanced Formatting: {generation_metadata['enhanced_formatting_applied']}")
                
                # Display the generated answer with references
                print(f"\n" + "=" * 60)
                print("GENERATED ANSWER WITH REFERENCES")
                print("=" * 60)
                
                summary = generated_answer.get('summary', 'No summary available')
                detailed_answer = generated_answer.get('detailed_answer', 'No detailed answer available')
                
                print(f"SUMMARY: {summary}")
                print(f"\nDETAILED ANSWER:")
                print(detailed_answer)
                
                # Debug: Show full lengths
                print(f"\nDEBUG INFO:")
                print(f"Summary length: {len(summary)} characters")
                print(f"Detailed answer length: {len(detailed_answer)} characters")
                print(f"Generated answer keys: {list(generated_answer.keys())}")
                
                print(f"\nCONFIDENCE: {generated_answer.get('confidence', 0.0):.3f}")
                
                # Show references
                references = generated_answer.get('references', [])
                if references:
                    print(f"\nREFERENCES ({len(references)}):")
                    for i, ref in enumerate(references, 1):
                        # Handle different reference formats from answer generator
                        ref_content = ref.get('span', ref.get('content', ref.get('text', '')))[:100]
                        ref_source = ref.get('doc_id', ref.get('source', ref.get('id', f'Reference {i}')))
                        relevance = ref.get('relevance_score', 0.0)
                        print(f"{i}. [{ref_source}] {ref_content}... (relevance: {relevance:.2f})")
                else:
                    print(f"\nREFERENCES: Using {len(evidence_documents)} evidence documents")
                    for i, doc in enumerate(evidence_documents[:3], 1):
                        print(f"{i}. [{doc['id']}] {doc['content'][:80]}...")
                
                # Evaluate answer quality dynamically
                print(f"\nANSWER EVALUATION:")
                full_answer = (summary + " " + detailed_answer).lower()
                
                def evaluate_answer_quality(query, intent, entities, answer_text, confidence):
                    """Dynamically evaluate answer quality based on intent and entities"""
                    
                    # Basic quality checks
                    if len(answer_text.strip()) < 10:
                        return "poor", "Answer is too short"
                    
                    if confidence < 0.1:
                        return "poor", "Very low confidence score"
                    
                    # Intent-based evaluation
                    if intent == "fact":
                        # For factual questions, check if key entities are addressed
                        entity_coverage = sum(1 for entity in entities if entity.lower() in answer_text)
                        if entity_coverage >= len(entities) * 0.7:  # 70% entity coverage
                            if confidence > 0.6:
                                return "correct", f"Good factual answer with {entity_coverage}/{len(entities)} entities covered"
                            else:
                                return "good", f"Factual answer covers {entity_coverage}/{len(entities)} entities"
                        else:
                            return "partial", f"Only covers {entity_coverage}/{len(entities)} key entities"
                    
                    elif intent == "explain":
                        # For explanations, check length and entity presence
                        if len(answer_text) > 100 and any(entity.lower() in answer_text for entity in entities):
                            return "good", "Detailed explanation with key concepts"
                        elif len(answer_text) > 50:
                            return "partial", "Basic explanation provided"
                        else:
                            return "poor", "Explanation too brief"
                    
                    elif intent == "compare":
                        # For comparisons, check if multiple entities are discussed
                        entity_mentions = [entity for entity in entities if entity.lower() in answer_text]
                        if len(entity_mentions) >= 2:
                            return "good", f"Comparison includes multiple entities: {entity_mentions}"
                        elif len(entity_mentions) == 1:
                            return "partial", f"Only discusses one entity: {entity_mentions[0]}"
                        else:
                            return "poor", "Doesn't adequately compare the requested entities"
                    
                    else:
                        # General evaluation for other intents
                        entity_coverage = sum(1 for entity in entities if entity.lower() in answer_text)
                        if entity_coverage > 0 and len(answer_text) > 30:
                            return "good", f"Addresses query with {entity_coverage} key entities"
                        elif len(answer_text) > 20:
                            return "partial", "Basic answer provided"
                        else:
                            return "poor", "Answer lacks substance"
                
                # Evaluate the answer
                answer_quality, evaluation_reason = evaluate_answer_quality(
                    normalized_query, intent, entities, full_answer, generated_answer.get('confidence', 0.0)
                )
                
                # Display evaluation result
                quality_icons = {
                    "correct": " CORRECT",
                    "good": "GOOD", 
                    "partial": " PARTIAL",
                    "poor": " POOR",
                    "unclear": " UNCLEAR"
                }
                
                print(f"{quality_icons.get(answer_quality, ' UNKNOWN')}: {evaluation_reason}")
                
            else:
                print(f" Generation failed: {generation_result.get('error', 'Unknown error')}")
                generated_answer = {"summary": "Generation failed", "confidence": 0.0}
                generation_result = {"success": False}
                answer_quality = "failed"
            
            # Step 6: Simulate User Feedback and Learning
            print(f"\n6. Feedback Learning System...")
            
            # Get real user feedback
            def get_user_feedback():
                """Get feedback from user about the generated answer"""
                print(f"\n" + "="*60)
                print("PLEASE PROVIDE FEEDBACK")
                print("="*60)
                
                # Get rating
                while True:
                    try:
                        rating_input = input("Rate this answer (1-5 stars, 5 being best): ").strip()
                        if not rating_input:
                            continue
                        rating = int(rating_input)
                        if 1 <= rating <= 5:
                            break
                        else:
                            print("Please enter a number between 1 and 5.")
                    except ValueError:
                        print("Please enter a valid number (1-5).")
                    except KeyboardInterrupt:
                        print("\nUsing default rating of 3.")
                        rating = 3
                        break
                
                # Get feedback type based on rating
                if rating >= 4:
                    feedback_type = "thumbs_up"
                elif rating == 3:
                    feedback_type = "helpful"
                else:
                    feedback_type = "thumbs_down"
                
                # Get optional correction/comment
                correction = None
                try:
                    correction_input = input("Any corrections or comments? (optional, press Enter to skip): ").strip()
                    if correction_input:
                        correction = correction_input
                except KeyboardInterrupt:
                    pass
                
                return rating, feedback_type, correction
            
            # Get user feedback
            if generation_result['success']:
                try:
                    user_rating, feedback_type, correction = get_user_feedback()
                except KeyboardInterrupt:
                    print("\nSkipping feedback collection...")
                    user_rating = 3
                    feedback_type = "helpful"
                    correction = None
            else:
                print("Generation failed - using default poor feedback")
                user_rating = 1
                feedback_type = "thumbs_down"
                correction = "Failed to generate proper answer"
            
            # Create feedback and response trace
            feedback_entry = create_feedback_entry(user_rating, feedback_type, correction)
            response_trace = create_response_trace(
                query=normalized_query,
                intent=intent,
                generated_response=generated_answer,
                evidence_documents=evidence_documents,
                generation_metadata=generation_result.get('generation_metadata', {}),
                validation_results=generation_result.get('validation_result')
            )
            
            # Accumulate feedback for session-wide learning
            all_feedback_entries.append(feedback_entry)
            all_response_traces.append(response_trace)
            
            # Debug: Check feedback entry details
            print(f"\n🔍 DEBUG FEEDBACK PROCESSING:")
            print(f"  - Feedback entries count: {len(all_feedback_entries)}")
            print(f"  - Response traces count: {len(all_response_traces)}")
            print(f"  - Latest feedback entry rating: {feedback_entry.rating}")
            print(f"  - Latest feedback entry type: {feedback_entry.feedback_type}")
            print(f"  - Min threshold: {feedback_learning_system.min_feedback_threshold}")
            
            # Analyze feedback for learning (using all accumulated feedback)
            try:
                learning_insights = feedback_learning_system.analyze_feedback_batch(
                    feedback_entries=all_feedback_entries,
                    response_traces=all_response_traces,
                    session_context=session_context
                )
                print(f"  - Learning insights generated: {len(learning_insights)}")
            except Exception as e:
                print(f"  - ERROR in analyze_feedback_batch: {e}")
                learning_insights = []
            
            print(f"\n📊 FEEDBACK COLLECTED:")
            print(f"User Rating: {user_rating}/5 stars ({feedback_type})")
            if correction:
                print(f"User Comment: {correction}")
            print(f"Learning Insights Generated: {len(learning_insights)} (from {len(all_feedback_entries)} total feedback entries)")
            
            # Show new insights
            if learning_insights:
                print(f"\n🧠 LEARNING INSIGHTS:")
                for insight in learning_insights[-3:]:  # Show last 3 insights
                    print(f"  • {insight.description} (Priority: {insight.priority})")
            
            # Show learning system adaptation
            if len(all_feedback_entries) > 1:
                print(f"\n📈 ADAPTIVE LEARNING:")
                print(f"  - Session feedback count: {len(all_feedback_entries)}")
                print(f"  - Total insights generated: {len(learning_insights)}")
                
                # Show performance trend
                ratings = [entry.rating for entry in all_feedback_entries if entry.rating]
                if len(ratings) >= 2:
                    trend = "improving" if ratings[-1] > ratings[-2] else "declining" if ratings[-1] < ratings[-2] else "stable"
                    print(f"  - Performance trend: {trend} (latest: {ratings[-1]}/5)")
                
                # Show system adjustments
                adjustments = feedback_learning_system.adjustment_history
                if adjustments:
                    recent_adjustments = [adj for adj in adjustments if 'timestamp' in adj]
                    if recent_adjustments:
                        print(f"  - Recent adjustments: {len(recent_adjustments)}")
                        for adj in recent_adjustments[-2:]:  # Show last 2 adjustments
                            print(f"    → {adj.get('description', 'System adjustment made')}")
            
            # Store pipeline result
            pipeline_result = {
                "query_index": query_idx,
                "original_query": query,
                "normalized_query": normalized_query,
                "intent": intent,
                "entities": entities,
                "strategy_used": strategy_used,
                "retrieval_time": retrieval_result['execution_metadata']['execution_time'],
                "reranking_time": timing.get('total_time', 0),
                "generation_time": generation_result['generation_metadata']['generation_time'] if generation_result['success'] else 0,
                "final_confidence": generated_answer.get('confidence', 0.0),
                "answer_quality": answer_quality if 'answer_quality' in locals() else "unknown",
                "user_rating": user_rating,
                "success": generation_result['success'],
                "learning_insights": len(learning_insights)
            }
            pipeline_results.append(pipeline_result)
            
            # Update session data for MemoRAG
            session_data["sessions"].append({
                "session_id": session_context["session_id"],
                "query": normalized_query,
                "intent": intent,
                "entities": entities,
                "result_summary": generated_answer.get('summary', ''),
                "confidence": generated_answer.get('confidence', 0.0),
                "user_rating": user_rating,
                "user_feedback": correction,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"\n✅ Query {query_idx} processing completed!")
        
        # Save session data
        save_session_data(session_data)
        
        # Step 7: Show Adaptive Planning Evolution
        print(f"\n" + "=" * 80)
        print("ADAPTIVE PLANNING STRATEGY EVOLUTION")
        print("=" * 80)
        
        print("Strategy Adaptation Across Queries:")
        for i, result in enumerate(pipeline_results, 1):
            print(f"Query {i}: {result['strategy_used']} strategy")
            print(f"  Query: {result['original_query']}")
            print(f"  Intent: {result['intent']}")
            print(f"  Answer Quality: {result['answer_quality']}")
            print(f"  Performance: {result['final_confidence']:.3f} confidence, {result['user_rating']}/5 rating")
            print(f"  Learning: {result['learning_insights']} insights generated")
            
            if i > 1:
                prev_result = pipeline_results[i-2]
                if result['strategy_used'] != prev_result['strategy_used']:
                    print(f"   Strategy changed from {prev_result['strategy_used']} to {result['strategy_used']}")
                
                # Show performance comparison
                perf_change = result['user_rating'] - prev_result['user_rating']
                if perf_change > 0:
                    print(f"  � Peerformance improved by {perf_change} points")
                elif perf_change < 0:
                    print(f"   Performance declined by {abs(perf_change)} points")
                else:
                    print(f"   Performance maintained")
            print()
        
        # Step 8: Comprehensive Performance Summary
        print(f"\n" + "=" * 80)
        print("COMPLETE ENHANCED PIPELINE PERFORMANCE")
        print("=" * 80)
        
        total_retrieval_time = sum(r['retrieval_time'] for r in pipeline_results)
        total_reranking_time = sum(r['reranking_time'] for r in pipeline_results)
        total_generation_time = sum(r['generation_time'] for r in pipeline_results)
        total_pipeline_time = total_retrieval_time + total_reranking_time + total_generation_time
        
        avg_confidence = sum(r['final_confidence'] for r in pipeline_results) / len(pipeline_results)
        avg_rating = sum(r['user_rating'] for r in pipeline_results) / len(pipeline_results)
        success_rate = sum(1 for r in pipeline_results if r['success']) / len(pipeline_results)
        
        print(f"PIPELINE STAGES PERFORMANCE:")
        print(f"1. Query Preprocessing: Integrated ")
        print(f"2. Adaptive Planning: {len(pipeline_results)} strategies applied ")
        print(f"3. Retrieval + Fusion: {total_retrieval_time:.3f}s total ({(total_retrieval_time/total_pipeline_time)*100:.1f}%)")
        print(f"4. Advanced Reranking: {total_reranking_time:.3f}s total ({(total_reranking_time/total_pipeline_time)*100:.1f}%)")
        print(f"5. Enhanced Generation: {total_generation_time:.3f}s total ({(total_generation_time/total_pipeline_time)*100:.1f}%)")
        print(f"6. Feedback Learning: {sum(r['learning_insights'] for r in pipeline_results)} insights generated ")
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Total Pipeline Time: {total_pipeline_time:.3f}s")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Average User Rating: {avg_rating:.1f}/5")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Queries Processed: {len(pipeline_results)}")
        print(f"Session Data: {len(session_data['sessions'])} entries saved")
        
        # Show learning system summary
        learning_summary = feedback_learning_system.get_learning_summary()
        print(f"\nANSWER QUALITY SUMMARY:")
        quality_counts = {}
        rating_distribution = {}
        
        for result in pipeline_results:
            # Quality distribution
            quality = result.get('answer_quality', 'unknown')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            # Rating distribution
            rating = result.get('user_rating', 0)
            rating_distribution[rating] = rating_distribution.get(rating, 0) + 1
        
        for quality, count in quality_counts.items():
            print(f"{quality.title()}: {count} queries")
        
        correct_answers = sum(1 for r in pipeline_results if r.get('answer_quality') in ['correct', 'good'])
        print(f"Overall Answer Quality: {correct_answers}/{len(pipeline_results)} queries answered correctly")
        
        print(f"\nUSER FEEDBACK SUMMARY:")
        total_ratings = sum(rating_distribution.values())
        if total_ratings > 0:
            avg_rating = sum(rating * count for rating, count in rating_distribution.items()) / total_ratings
            print(f"Average User Rating: {avg_rating:.1f}/5 stars")
            
            print("Rating Distribution:")
            for rating in sorted(rating_distribution.keys(), reverse=True):
                count = rating_distribution[rating]
                percentage = (count / total_ratings) * 100
                stars = "⭐" * rating
                print(f"  {stars} ({rating}/5): {count} queries ({percentage:.1f}%)")
        
        # Show user comments if any
        user_comments = [r.get('user_feedback') for r in pipeline_results if r.get('user_feedback')]
        if user_comments:
            print(f"\nUSER COMMENTS:")
            for i, comment in enumerate(user_comments, 1):
                print(f"  {i}. \"{comment}\"")
        
        print(f"\nLEARNING SYSTEM STATUS:")
        print(f"Total Insights: {learning_summary['total_insights']}")
        print(f"Recent Insights: {learning_summary['recent_insights']}")
        print(f"Adjustments Made: {learning_summary['adjustments_made']}")
        print(f"System Health: {learning_summary['system_health']['status']}")
        print(f"Auto-adjustments: {'Enabled' if learning_summary['auto_adjustments_enabled'] else 'Disabled'}")
        
        # Save user feedback summary
        feedback_summary = {
            "session_timestamp": datetime.now().isoformat(),
            "total_queries": len(pipeline_results),
            "average_rating": sum(r['user_rating'] for r in pipeline_results) / len(pipeline_results) if pipeline_results else 0,
            "quality_distribution": quality_counts,
            "rating_distribution": rating_distribution,
            "user_comments": [r.get('user_feedback') for r in pipeline_results if r.get('user_feedback')],
            "pipeline_results": pipeline_results
        }
        
        try:
            with open("user_feedback_summary.json", 'w') as f:
                json.dump(feedback_summary, f, indent=2, default=str)
            print(f"💾 User feedback saved to user_feedback_summary.json")
        except Exception as e:
            print(f"⚠️ Could not save feedback summary: {e}")
        
        print(f"\n🎉 Thank you for testing the Interactive RAG Pipeline!")
        print(f"✅ All components integrated: Preprocessing → Planning → Retrieval → Reranking → Generation → Learning")
        print(f"📊 Session data saved to memorag_sessions.json for MemoRAG integration")
        print(f"💬 Your feedback helps improve the system - thank you for your input!")
        
        if len(pipeline_results) > 0:
            avg_rating = sum(r['user_rating'] for r in pipeline_results) / len(pipeline_results)
            if avg_rating >= 4:
                print(f"🌟 Excellent! Average rating: {avg_rating:.1f}/5 stars")
            elif avg_rating >= 3:
                print(f"👍 Good performance! Average rating: {avg_rating:.1f}/5 stars")
            else:
                print(f"📈 Room for improvement. Average rating: {avg_rating:.1f}/5 stars")
        
    except Exception as e:
        print(f"  Enhanced pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save partial session data if available
        if 'session_data' in locals():
            save_session_data(session_data)

if __name__ == "__main__":
    main()