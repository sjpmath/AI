# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import proto_sample_pb2 as proto__sample__pb2


class AI_OnlineClassMonitorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.process = channel.unary_unary(
                '/AI_OnlineClassMonitor/process',
                request_serializer=proto__sample__pb2.InferenceRequest.SerializeToString,
                response_deserializer=proto__sample__pb2.InferenceReply.FromString,
                )


class AI_OnlineClassMonitorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def process(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_AI_OnlineClassMonitorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'process': grpc.unary_unary_rpc_method_handler(
                    servicer.process,
                    request_deserializer=proto__sample__pb2.InferenceRequest.FromString,
                    response_serializer=proto__sample__pb2.InferenceReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'AI_OnlineClassMonitor', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class AI_OnlineClassMonitor(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def process(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/AI_OnlineClassMonitor/process',
            proto__sample__pb2.InferenceRequest.SerializeToString,
            proto__sample__pb2.InferenceReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
