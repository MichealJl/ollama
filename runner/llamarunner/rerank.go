package llamarunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/ollama/ollama/pkg"
	"log/slog"
	"net/http"
	"time"
)

type RerankRequest struct {
	Model       string   `json:"model"`
	Query       string   `json:"query"`
	TopN        int      `json:"top_n"`     // return top N documents
	Documents   []string `json:"documents"` // list of documents to rerank
	CachePrompt bool     `json:"cache_prompt"`
}

type RerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float32 `json:"relevance_score"`
	} `json:"results"`
}

func (s *Server) rerank(w http.ResponseWriter, r *http.Request) {
	defer pkg.Timing("runner-rerank", time.Now())
	var req RerankRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad rereank request: %s", err), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")

	var rsp RerankResponse
	rsp.Results = make([]struct {
		Index          int     `json:"index"`
		RelevanceScore float32 `json:"relevance_score"`
	}, len(req.Documents))

	for i, doc := range req.Documents {
		// reranking prompt format: [BOS]query[EOS][SEP]doc[EOS]
		p := ""
		if !s.model.AddBOSToken() {
			p += s.model.TokenToPiece(int(s.lc.GetTokenBOS()))
		}
		p += req.Query + s.model.TokenToPiece(int(s.lc.GetTokenEOS())) + s.model.TokenToPiece(int(s.lc.GetTokenSEP())) + doc
		if !s.model.AddEOSToken() {
			p += s.model.TokenToPiece(int(s.lc.GetTokenEOS()))
		}
		seq, err := s.NewSequence(p, nil, NewSequenceParams{embedding: true})
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
			return
		}

		// Ensure there is a place to put the sequence, released when removed from s.seqs
		if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
			if errors.Is(err, context.Canceled) {
				slog.Info("aborting reranking request due to client closing the connection")
			} else {
				slog.Error("Failed to acquire semaphore", "error", err)
			}
			return
		}

		s.mu.Lock()
		found := false
		for i, sq := range s.seqs {
			if sq == nil {
				seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs, req.CachePrompt)
				if err != nil {
					s.mu.Unlock()
					http.Error(w, fmt.Sprintf("Failed to load cache: %v", err), http.StatusInternalServerError)
					return
				}
				s.seqs[i] = seq
				s.cond.Signal()
				found = true
				break
			}
		}
		s.mu.Unlock()

		if !found {
			http.Error(w, "could not find an available sequence", http.StatusInternalServerError)
			return
		}

		score := <-seq.embedding
		rsp.Results[i].Index = i
		rsp.Results[i].RelevanceScore = score[0]
	}
	if err := json.NewEncoder(w).Encode(&rsp); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}
