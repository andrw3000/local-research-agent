import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "./components/ActivityTimeline";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { ChatMessagesView } from "./components/ChatMessagesView";

// Convert [REF]source[/REF] to markdown links with both anchor and URL references
const convertReferencesToMarkdown = (text: string): string => {
  const references = new Map<string, number>();

  return text.replace(
    /\[REF\](.*?)(?:\|(.*?))?\[\/REF\]/g,
    (_, source, url) => {
      // Get or create reference number
      let refNumber = references.get(source);
      if (refNumber === undefined) {
        refNumber = references.size + 1;
        references.set(source, refNumber);
      }

      // Create a URL-friendly version of the source for the anchor
      const anchor = source.toLowerCase().replace(/\s+/g, "-");
      const urlPart = url
        ? `<a href="${url}" target="_blank" rel="noopener noreferrer">[${refNumber}]</a>`
        : "";

      return `[${source}](#${anchor})${urlPart}`;
    }
  );
};

interface ChatMessage {
  role: "user" | "ai";
  content: string;
  id?: string;
}

interface Message {
  type: "human" | "ai";
  content: string;
  id: string;
}

export default function App() {
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const apiUrl = import.meta.env.DEV
    ? "http://localhost:2024"
    : "http://localhost:8123";

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [messages]);

  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !isLoading &&
      messages.length > 0
    ) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
      setProcessedEventsTimeline([]);
    }
  }, [messages, isLoading, processedEventsTimeline]);

  const handleSubmit = useCallback(
    async (submittedInputValue: string, effort: string, model: string) => {
      if (!submittedInputValue.trim()) return;
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;
      setIsLoading(true);

      let initial_search_query_count = 3;
      let max_research_loops = 3;
      switch (effort) {
        case "low":
          initial_search_query_count = 1;
          max_research_loops = 1;
          break;
        case "high":
          initial_search_query_count = 5;
          max_research_loops = 10;
          break;
      }

      // Add new message to UI immediately
      const newMessage: Message = {
        type: "human",
        content: submittedInputValue,
        id: Date.now().toString(),
      };
      const updatedMessages = [...messages, newMessage];
      setMessages(updatedMessages);

      // Convert messages to backend format
      const backendMessages: ChatMessage[] = updatedMessages.map((msg) => ({
        role: msg.type === "human" ? "user" : "ai",
        content:
          typeof msg.content === "string"
            ? msg.content
            : JSON.stringify(msg.content),
        id: msg.id,
      }));

      try {
        abortControllerRef.current = new AbortController();

        const response = await fetch(`${apiUrl}/app/agent`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            messages: backendMessages,
            initial_search_query_count,
            max_research_loops,
            ollama_llm: model,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("No response body");
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");

          // Process all complete lines
          for (let i = 0; i < lines.length - 1; i++) {
            const line = lines[i].trim();
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));

                if (data.data?.messages) {
                  // Handle final messages update
                  const newMessages = data.data.messages.map(
                    (msg: ChatMessage) => ({
                      type: msg.role === "user" ? "human" : "ai",
                      content:
                        msg.role === "ai"
                          ? convertReferencesToMarkdown(msg.content)
                          : msg.content,
                      id: msg.id || Date.now().toString(),
                    })
                  );
                  setMessages(newMessages);
                  hasFinalizeEventOccurredRef.current = true;
                } else if (data.generate_query || data.research) {
                  // Handle progress events
                  const event = data.generate_query || data.research;
                  setProcessedEventsTimeline((prev) => [
                    ...prev,
                    {
                      title: data.generate_query
                        ? "Generating Search Queries"
                        : "Research Progress",
                      data: event.status,
                    },
                  ]);
                } else if (data.error) {
                  // Handle error events
                  setProcessedEventsTimeline((prev) => [
                    ...prev,
                    {
                      title: "Error",
                      data: data.error.status,
                    },
                  ]);
                }
              } catch (e) {
                console.error("Error parsing SSE data:", e);
              }
            }
          }
          // Keep the last incomplete line in the buffer
          buffer = lines[lines.length - 1];
        }
      } catch (err: unknown) {
        const error = err as Error;
        if (error.name === "AbortError") {
          console.log("Request was cancelled");
        } else {
          setProcessedEventsTimeline((prev) => [
            ...prev,
            {
              title: "Error",
              data: error.message,
            },
          ]);
        }
      } finally {
        setIsLoading(false);
      }
    },
    [messages]
  );

  const handleCancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  }, []);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="flex-1 flex flex-col overflow-hidden max-w-4xl mx-auto w-full">
        <div
          className={`flex-1 overflow-y-auto ${
            messages.length === 0 ? "flex" : ""
          }`}
        >
          {messages.length === 0 ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={isLoading}
              onCancel={handleCancel}
            />
          ) : (
            <ChatMessagesView
              messages={messages}
              isLoading={isLoading}
              scrollAreaRef={scrollAreaRef}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
              liveActivityEvents={processedEventsTimeline}
              historicalActivities={historicalActivities}
            />
          )}
        </div>
      </main>
    </div>
  );
}
