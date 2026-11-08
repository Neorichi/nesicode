import type { Attachment } from './api'

/**
 * Unsent drafts per conversation (text + its attachments, which live in that conversation's cage), kept in memory
 * for the session: leaving a conversation and coming back finds the draft; another conversation never sees it.
 */
export type Draft = { text: string; atts: Attachment[]; threadId?: number }

export const drafts = new Map<string, Draft>()

/** The conversation is gone (its id may be reused): drop its draft. */
export function forgetDraft(key: string) {
  drafts.delete(key)
}
