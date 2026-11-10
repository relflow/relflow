-- Raw structure: the entity row, its related rows nested by recency, and their looked-up context.
-- No counts, windows, or summaries are computed here; ages measure each timestamp against the anchor.
WITH
anchors AS (
    SELECT "__row_position", "__entity_key", "__as_of"
    FROM read_parquet($anchors)
    WHERE "__row_position" >= $start AND "__row_position" < $stop
),
root AS (
    SELECT anchor."__row_position", anchor."__entity_key", anchor."__as_of",
        anchor."__as_of" AS "as_of",
        root."CreationDate" AS "created_at",
        date_diff('microseconds', root."CreationDate", anchor."__as_of") / 86400000000.0 AS "created_at_age_days",
        root."Location" AS "location",
        root."AboutMe" AS "about_me"
    FROM anchors AS anchor
    LEFT JOIN "users" AS root ON anchor."__entity_key" = root."Id" AND root."CreationDate" <= anchor."__as_of"
    QUALIFY row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY root."CreationDate" DESC NULLS LAST, root."__source_row_id" DESC NULLS LAST) = 1
),
badges_events AS (
    SELECT parent."__row_position", parent."__as_of",
        event."Date" AS "__event_time", event."__source_row_id" AS "__source_order",
        event."Date" AS "occurred_at",
        date_diff('microseconds', event."Date", parent."__as_of") / 86400000000.0 AS "age_days",
        event."Name" AS "badge_name",
        event."Class" AS "badge_class",
        event."TagBased" AS "tag_based"
    FROM root AS parent
    INNER JOIN "badges" AS event ON event."UserId" = parent."__entity_key" AND event."Date" <= parent."__as_of"
),
badges_ranked AS (
    SELECT *, row_number() OVER (PARTITION BY "__row_position" ORDER BY "__event_time" DESC NULLS LAST, "__source_order" DESC) AS "__recency"
    FROM badges_events
),
badges_grouped AS (
    SELECT badges_ranked."__row_position",
        list(struct_pack(
            "occurred_at" := badges_ranked."occurred_at",
            "age_days" := badges_ranked."age_days",
            "badge_name" := badges_ranked."badge_name",
            "badge_class" := badges_ranked."badge_class",
            "tag_based" := badges_ranked."tag_based"
        ) ORDER BY badges_ranked."__event_time" ASC NULLS LAST, badges_ranked."__source_order" ASC) AS "badges"
    FROM badges_ranked
    WHERE badges_ranked."__recency" <= 64
    GROUP BY badges_ranked."__row_position"
),
posts_events AS (
    SELECT parent."__row_position", parent."__as_of",
        event."CreationDate" AS "__event_time", event."__source_row_id" AS "__source_order",
        event."Id" AS "__nested_key",
        event."CreationDate" AS "occurred_at",
        date_diff('microseconds', event."CreationDate", parent."__as_of") / 86400000000.0 AS "age_days",
        event."PostTypeId" AS "post_type",
        CASE WHEN event."ParentId" IS NULL THEN NULL ELSE 'rel-stack:post:' || CAST(event."ParentId" AS VARCHAR) END AS "parent_id",
        event."Title" AS "title",
        event."Body" AS "body",
        event."Tags" AS "tags"
    FROM root AS parent
    INNER JOIN "posts" AS event ON event."OwnerUserId" = parent."__entity_key" AND event."CreationDate" <= parent."__as_of"
),
posts_ranked AS (
    SELECT *, row_number() OVER (PARTITION BY "__row_position" ORDER BY "__event_time" DESC NULLS LAST, "__source_order" DESC) AS "__recency"
    FROM posts_events
),
posts_events_keys AS (
    SELECT "__row_position", "__as_of", "__nested_key" AS "__child_key" FROM posts_ranked WHERE "__recency" <= 64
),
posts_votes_events AS (
    SELECT parent."__row_position", parent."__child_key", parent."__as_of",
        event."CreationDate" AS "__event_time", event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "occurred_at",
        date_diff('microseconds', event."CreationDate", parent."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."UserId" IS NULL THEN NULL ELSE 'rel-stack:user:' || CAST(event."UserId" AS VARCHAR) END AS "user_id",
        event."VoteTypeId" AS "vote_type"
    FROM posts_events_keys AS parent
    INNER JOIN "votes" AS event ON event."PostId" = parent."__child_key" AND event."CreationDate" <= parent."__as_of"
),
posts_votes_ranked AS (
    SELECT *, row_number() OVER (PARTITION BY "__row_position", "__child_key" ORDER BY "__event_time" DESC NULLS LAST, "__source_order" DESC) AS "__recency"
    FROM posts_votes_events
),
posts_votes_grouped AS (
    SELECT posts_votes_ranked."__row_position", posts_votes_ranked."__child_key",
        list(struct_pack(
            "occurred_at" := posts_votes_ranked."occurred_at",
            "age_days" := posts_votes_ranked."age_days",
            "user_id" := posts_votes_ranked."user_id",
            "vote_type" := posts_votes_ranked."vote_type"
        ) ORDER BY posts_votes_ranked."__event_time" ASC NULLS LAST, posts_votes_ranked."__source_order" ASC) AS "votes"
    FROM posts_votes_ranked
    WHERE posts_votes_ranked."__recency" <= 16
    GROUP BY posts_votes_ranked."__row_position", posts_votes_ranked."__child_key"
),
posts_comments_events AS (
    SELECT parent."__row_position", parent."__child_key", parent."__as_of",
        event."CreationDate" AS "__event_time", event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "occurred_at",
        date_diff('microseconds', event."CreationDate", parent."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."UserId" IS NULL THEN NULL ELSE 'rel-stack:user:' || CAST(event."UserId" AS VARCHAR) END AS "user_id",
        event."Text" AS "comment_text"
    FROM posts_events_keys AS parent
    INNER JOIN "comments" AS event ON event."PostId" = parent."__child_key" AND event."CreationDate" <= parent."__as_of"
),
posts_comments_ranked AS (
    SELECT *, row_number() OVER (PARTITION BY "__row_position", "__child_key" ORDER BY "__event_time" DESC NULLS LAST, "__source_order" DESC) AS "__recency"
    FROM posts_comments_events
),
posts_comments_grouped AS (
    SELECT posts_comments_ranked."__row_position", posts_comments_ranked."__child_key",
        list(struct_pack(
            "occurred_at" := posts_comments_ranked."occurred_at",
            "age_days" := posts_comments_ranked."age_days",
            "user_id" := posts_comments_ranked."user_id",
            "comment_text" := posts_comments_ranked."comment_text"
        ) ORDER BY posts_comments_ranked."__event_time" ASC NULLS LAST, posts_comments_ranked."__source_order" ASC) AS "comments"
    FROM posts_comments_ranked
    WHERE posts_comments_ranked."__recency" <= 8
    GROUP BY posts_comments_ranked."__row_position", posts_comments_ranked."__child_key"
),
posts_grouped AS (
    SELECT posts_ranked."__row_position",
        list(struct_pack(
            "occurred_at" := posts_ranked."occurred_at",
            "age_days" := posts_ranked."age_days",
            "post_type" := posts_ranked."post_type",
            "parent_id" := posts_ranked."parent_id",
            "title" := posts_ranked."title",
            "body" := posts_ranked."body",
            "tags" := posts_ranked."tags",
            "votes" := coalesce(posts_votes_grouped."votes", []),
            "comments" := coalesce(posts_comments_grouped."comments", [])
        ) ORDER BY posts_ranked."__event_time" ASC NULLS LAST, posts_ranked."__source_order" ASC) AS "posts"
    FROM posts_ranked
    LEFT JOIN posts_votes_grouped ON posts_votes_grouped."__row_position" = posts_ranked."__row_position" AND posts_votes_grouped."__child_key" = posts_ranked."__nested_key"
    LEFT JOIN posts_comments_grouped ON posts_comments_grouped."__row_position" = posts_ranked."__row_position" AND posts_comments_grouped."__child_key" = posts_ranked."__nested_key"
    WHERE posts_ranked."__recency" <= 64
    GROUP BY posts_ranked."__row_position"
),
comments_events AS (
    SELECT parent."__row_position", parent."__as_of",
        event."CreationDate" AS "__event_time", event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "occurred_at",
        date_diff('microseconds', event."CreationDate", parent."__as_of") / 86400000000.0 AS "age_days",
        event."Text" AS "comment_text",
        post."CreationDate" AS "post_created_at",
        date_diff('microseconds', post."CreationDate", parent."__as_of") / 86400000000.0 AS "post_created_at_age_days",
        post."PostTypeId" AS "post_post_type",
        CASE WHEN post."OwnerUserId" IS NULL THEN NULL ELSE 'rel-stack:user:' || CAST(post."OwnerUserId" AS VARCHAR) END AS "post_owner_user_id",
        post."Title" AS "post_title",
        post."Body" AS "post_body",
        post."Tags" AS "post_tags"
    FROM root AS parent
    INNER JOIN "comments" AS event ON event."UserId" = parent."__entity_key" AND event."CreationDate" <= parent."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "posts" AS linked WHERE linked."Id" = event."PostId" AND linked."CreationDate" <= parent."__as_of" ORDER BY linked."CreationDate" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS post ON TRUE
),
comments_ranked AS (
    SELECT *, row_number() OVER (PARTITION BY "__row_position" ORDER BY "__event_time" DESC NULLS LAST, "__source_order" DESC) AS "__recency"
    FROM comments_events
),
comments_grouped AS (
    SELECT comments_ranked."__row_position",
        list(struct_pack(
            "occurred_at" := comments_ranked."occurred_at",
            "age_days" := comments_ranked."age_days",
            "comment_text" := comments_ranked."comment_text",
            "post_created_at" := comments_ranked."post_created_at",
            "post_created_at_age_days" := comments_ranked."post_created_at_age_days",
            "post_post_type" := comments_ranked."post_post_type",
            "post_owner_user_id" := comments_ranked."post_owner_user_id",
            "post_title" := comments_ranked."post_title",
            "post_body" := comments_ranked."post_body",
            "post_tags" := comments_ranked."post_tags"
        ) ORDER BY comments_ranked."__event_time" ASC NULLS LAST, comments_ranked."__source_order" ASC) AS "comments"
    FROM comments_ranked
    WHERE comments_ranked."__recency" <= 64
    GROUP BY comments_ranked."__row_position"
),
votes_events AS (
    SELECT parent."__row_position", parent."__as_of",
        event."CreationDate" AS "__event_time", event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "occurred_at",
        date_diff('microseconds', event."CreationDate", parent."__as_of") / 86400000000.0 AS "age_days",
        event."VoteTypeId" AS "vote_type",
        post."CreationDate" AS "post_created_at",
        date_diff('microseconds', post."CreationDate", parent."__as_of") / 86400000000.0 AS "post_created_at_age_days",
        post."PostTypeId" AS "post_post_type",
        CASE WHEN post."OwnerUserId" IS NULL THEN NULL ELSE 'rel-stack:user:' || CAST(post."OwnerUserId" AS VARCHAR) END AS "post_owner_user_id",
        post."Title" AS "post_title",
        post."Body" AS "post_body",
        post."Tags" AS "post_tags"
    FROM root AS parent
    INNER JOIN "votes" AS event ON event."UserId" = parent."__entity_key" AND event."CreationDate" <= parent."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "posts" AS linked WHERE linked."Id" = event."PostId" AND linked."CreationDate" <= parent."__as_of" ORDER BY linked."CreationDate" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS post ON TRUE
),
votes_ranked AS (
    SELECT *, row_number() OVER (PARTITION BY "__row_position" ORDER BY "__event_time" DESC NULLS LAST, "__source_order" DESC) AS "__recency"
    FROM votes_events
),
votes_grouped AS (
    SELECT votes_ranked."__row_position",
        list(struct_pack(
            "occurred_at" := votes_ranked."occurred_at",
            "age_days" := votes_ranked."age_days",
            "vote_type" := votes_ranked."vote_type",
            "post_created_at" := votes_ranked."post_created_at",
            "post_created_at_age_days" := votes_ranked."post_created_at_age_days",
            "post_post_type" := votes_ranked."post_post_type",
            "post_owner_user_id" := votes_ranked."post_owner_user_id",
            "post_title" := votes_ranked."post_title",
            "post_body" := votes_ranked."post_body",
            "post_tags" := votes_ranked."post_tags"
        ) ORDER BY votes_ranked."__event_time" ASC NULLS LAST, votes_ranked."__source_order" ASC) AS "votes"
    FROM votes_ranked
    WHERE votes_ranked."__recency" <= 64
    GROUP BY votes_ranked."__row_position"
),
edits_events AS (
    SELECT parent."__row_position", parent."__as_of",
        event."CreationDate" AS "__event_time", event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "occurred_at",
        date_diff('microseconds', event."CreationDate", parent."__as_of") / 86400000000.0 AS "age_days",
        event."PostHistoryTypeId" AS "revision_type",
        event."Comment" AS "revision_comment",
        post."CreationDate" AS "post_created_at",
        date_diff('microseconds', post."CreationDate", parent."__as_of") / 86400000000.0 AS "post_created_at_age_days",
        post."PostTypeId" AS "post_post_type",
        CASE WHEN post."OwnerUserId" IS NULL THEN NULL ELSE 'rel-stack:user:' || CAST(post."OwnerUserId" AS VARCHAR) END AS "post_owner_user_id",
        post."Title" AS "post_title",
        post."Body" AS "post_body",
        post."Tags" AS "post_tags"
    FROM root AS parent
    INNER JOIN "postHistory" AS event ON event."UserId" = parent."__entity_key" AND event."CreationDate" <= parent."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "posts" AS linked WHERE linked."Id" = event."PostId" AND linked."CreationDate" <= parent."__as_of" ORDER BY linked."CreationDate" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS post ON TRUE
),
edits_ranked AS (
    SELECT *, row_number() OVER (PARTITION BY "__row_position" ORDER BY "__event_time" DESC NULLS LAST, "__source_order" DESC) AS "__recency"
    FROM edits_events
),
edits_grouped AS (
    SELECT edits_ranked."__row_position",
        list(struct_pack(
            "occurred_at" := edits_ranked."occurred_at",
            "age_days" := edits_ranked."age_days",
            "revision_type" := edits_ranked."revision_type",
            "revision_comment" := edits_ranked."revision_comment",
            "post_created_at" := edits_ranked."post_created_at",
            "post_created_at_age_days" := edits_ranked."post_created_at_age_days",
            "post_post_type" := edits_ranked."post_post_type",
            "post_owner_user_id" := edits_ranked."post_owner_user_id",
            "post_title" := edits_ranked."post_title",
            "post_body" := edits_ranked."post_body",
            "post_tags" := edits_ranked."post_tags"
        ) ORDER BY edits_ranked."__event_time" ASC NULLS LAST, edits_ranked."__source_order" ASC) AS "edits"
    FROM edits_ranked
    WHERE edits_ranked."__recency" <= 64
    GROUP BY edits_ranked."__row_position"
)
SELECT
    root."__row_position",
    root."as_of",
    root."created_at",
    root."created_at_age_days",
    root."location",
    root."about_me",
    coalesce(badges_grouped."badges", []) AS "badges",
    coalesce(posts_grouped."posts", []) AS "posts",
    coalesce(comments_grouped."comments", []) AS "comments",
    coalesce(votes_grouped."votes", []) AS "votes",
    coalesce(edits_grouped."edits", []) AS "edits"
FROM root
LEFT JOIN badges_grouped USING ("__row_position")
LEFT JOIN posts_grouped USING ("__row_position")
LEFT JOIN comments_grouped USING ("__row_position")
LEFT JOIN votes_grouped USING ("__row_position")
LEFT JOIN edits_grouped USING ("__row_position")
ORDER BY root."__row_position"
