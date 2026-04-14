# Pipeline: test-monday-hello-lark

## Summary

A test pipeline that sends a Lark IM "hello" message every Monday at 9:00 AM UTC.

## Details

| Field | Value |
|-------|-------|
| **Pipeline ID** | `44f91dc2` |
| **Name** | `test-monday-hello-lark` |
| **Cron Expression** | `0 9 * * 1` (Every Monday at 9am UTC) |
| **Status** | Active |
| **Next Run** | 2026-04-21T09:00:00+00:00 |
| **Created At** | 2026-04-14T14:37:45 UTC |

## Workflow

Single-node pipeline with one `deliver` step:

```json
{
  "nodes": [
    {
      "id": "send_hello",
      "type": "deliver",
      "params": {
        "channel": "lark_im",
        "text": "Hello! This is your weekly Monday morning test message from the Vibe Pipeline. Have a great week!"
      }
    }
  ]
}
```

## Verification

- Pipeline registered and confirmed active via GET /api/schedules
- Test Lark IM message sent successfully (messageId: `om_x100b52d15b9b8938e2d3a3623cf26e3`)
- Entity logged via Python tool (job_id: `6ebd35e3`)
