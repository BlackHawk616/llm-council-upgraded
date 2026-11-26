import ReactMarkdown from 'react-markdown';
import './Stage3.css';

export default function Stage3({ finalResponse }) {
  if (!finalResponse) {
    return null;
  }

  const chairmanName = finalResponse.model.split('/')[1] || finalResponse.model;
  const selectedModel = finalResponse.selected_model;
  const selectedModelName = selectedModel
    ? (selectedModel.split('/')[1] || selectedModel)
    : null;
  const selectedAverageRank = finalResponse.selected_average_rank;

  return (
    <div className="stage stage3">
      <h3 className="stage-title">Stage 3: Final Council Answer</h3>
      <p className="stage3-description">
        Stage 3 is synthesized by an independent chairman model (non-voting).
        It can be different from the models shown in Stage 1 and Stage 2.
      </p>
      <div className="final-response">
        <div className="chairman-label">
          Independent Chairman: {chairmanName}
        </div>
        {selectedModelName && (
          <div className="selected-model-label">
            Best-Ranked Stage 2 Model: {selectedModelName} ({selectedModel})
            {typeof selectedAverageRank === 'number' && ` (Avg Rank: ${selectedAverageRank.toFixed(2)})`}
          </div>
        )}
        <div className="final-text markdown-content">
          <ReactMarkdown>{finalResponse.response}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
