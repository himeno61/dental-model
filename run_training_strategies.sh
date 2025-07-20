#!/bin/bash
# Simple training script with basic strategies

show_help() {
    echo "Usage: $0 [STRATEGY] [OPTIONS]"
    echo ""
    echo "Strategies:"
    echo "  test     - Quick 5-epoch test run"
    echo "  train    - Full training"
    echo "  resume   - Resume from checkpoint"
    echo "  parallel - Run multiple configs in parallel"
    echo ""
    echo "Options:"
    echo "  --config CONFIG_FILE  - Config to use (default: configs/default.yaml)"
    echo "  --device DEVICE       - Device to use (cuda/cpu, default: auto)"
    echo "  --resume CHECKPOINT   - Checkpoint path for resume"
    echo "  --help               - Show this help"
}

# Defaults
STRATEGY=""
CONFIG="configs/default.yaml"
DEVICE="auto"
RESUME_CHECKPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        test|train|resume|parallel)
            STRATEGY="$1"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [ -z "$STRATEGY" ]; then
    echo "Error: No strategy specified"
    show_help
    exit 1
fi

# Auto-detect device
if [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi

echo "Running: $STRATEGY with config: $CONFIG on device: $DEVICE"

case $STRATEGY in
    test)
        echo "Quick test (5 epochs)..."
        TMP_CONFIG="/tmp/test_config.yaml"
        cp "$CONFIG" "$TMP_CONFIG"
        sed -i.bak 's/epochs: [0-9]*/epochs: 5/' "$TMP_CONFIG"
        python -m src.train --config "$TMP_CONFIG" --device "$DEVICE"
        rm "$TMP_CONFIG" "$TMP_CONFIG.bak" 2>/dev/null
        ;;
    
    train)
        echo "Starting full training..."
        python -m src.train --config "$CONFIG" --device "$DEVICE"
        ;;
    
    resume)
        echo "Resuming training..."
        if [ -n "$RESUME_CHECKPOINT" ]; then
            python -m src.train --config "$CONFIG" --device "$DEVICE" --resume "$RESUME_CHECKPOINT"
        elif [ -f "outputs/best.pt" ]; then
            python -m src.train --config "$CONFIG" --device "$DEVICE" --resume outputs/best.pt
        else
            echo "Error: No checkpoint found"
            exit 1
        fi
        ;;
    
    parallel)
        echo "Running parallel experiments..."
        mkdir -p outputs/exp_default outputs/exp1
        
        python -m src.train --config configs/default.yaml --device "$DEVICE" &
        if [ -f "configs/experiment1.yaml" ]; then
            python -m src.train --config configs/experiment1.yaml --device "$DEVICE" &
        fi
        
        wait
        echo "All experiments completed!"
        ;;
    
    *)
        echo "Unknown strategy: $STRATEGY"
        exit 1
        ;;
esac
