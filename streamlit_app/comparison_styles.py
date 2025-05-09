"""
CSS styles for the visit comparison feature.
This file contains all the CSS needed for the enhanced visit comparison UI.
"""

def get_comparison_styles():
    """Return the CSS styles for the visit comparison feature."""
    return """
    <style>
        /* Color schemes for highlighting changes */
        .highlight-improvement {
            background-color: #d4edda;
            color: #155724;
            padding: 2px 5px;
            border-radius: 3px;
            font-weight: bold;
        }
        
        .highlight-decline {
            background-color: #f8d7da;
            color: #721c24;
            padding: 2px 5px;
            border-radius: 3px;
            font-weight: bold;
        }
        
        .highlight-same {
            background-color: #e2e3e5;
            color: #383d41;
            padding: 2px 5px;
            border-radius: 3px;
        }
        
        /* Header styling */
        .visit-header {
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        
        /* Change indicators */
        .change-arrow-up {
            color: #28a745;
            font-size: 1.3rem;
            margin: 0 5px;
        }
        
        .change-arrow-down {
            color: #dc3545;
            font-size: 1.3rem;
            margin: 0 5px;
        }
        
        .change-arrow-same {
            color: #6c757d;
            font-size: 1.3rem;
            margin: 0 5px;
        }
        
        /* Comparison table styling */
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .comparison-table th {
            background-color: #f8f9fa;
            color: #333;
            font-weight: bold;
            padding: 12px 15px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
        }
        
        .comparison-table td {
            padding: 10px 15px;
            border-bottom: 1px solid #eee;
        }
        
        .comparison-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .comparison-table tr:hover {
            background-color: #f1f1f1;
        }
        
        /* Category headers */
        .category-header {
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-left: 10px;
            border-left: 4px solid #4e73df;
            color: #333;
        }
        
        /* Summary box */
        .summary-box {
            background-color: #f8f9fa;
            border-left: 5px solid #4e73df;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        /* Changes summary card */
        .changes-card {
            display: flex;
            flex-direction: column;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 15px 0;
            overflow: hidden;
        }
        
        .changes-card-header {
            background-color: #4e73df;
            color: white;
            padding: 10px 15px;
            font-weight: bold;
        }
        
        .changes-card-body {
            padding: 15px;
        }
        
        .change-stat {
            display: inline-block;
            margin-right: 20px;
            text-align: center;
        }
        
        .change-stat-value {
            font-size: 1.8rem;
            font-weight: bold;
        }
        
        .change-stat-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
        
        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Print-friendly table */
        @media print {
            .comparison-table {
                width: 100%;
                border-collapse: collapse;
            }
            
            .comparison-table th, 
            .comparison-table td {
                border: 1px solid #ddd;
                padding: 8px;
            }
            
            .highlight-improvement,
            .highlight-decline,
            .highlight-same {
                padding: 2px 5px;
                border-radius: 0;
            }
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .changes-card-body {
                flex-direction: column;
            }
            
            .change-stat {
                margin-bottom: 15px;
            }
        }
    </style>
    """ 